"""PaperLens Web API。"""

from contextlib import asynccontextmanager
from contextvars import ContextVar
import hashlib
import logging
from pathlib import Path
import shutil
import time
import uuid
import asyncio
import re

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.agent.router import QueryRouter
from app.domain import citation_payload, sentence_evidence
from app.repository import PaperRepository
from app.services.parser import PaperParser
from app.services.reading import ReadingService
from app.services.retrieval import HybridRetriever, INDEX_VERSION
from app.multimodal import FigureUnderstandingService
from app.evaluation import evaluate_records
from app.multi_agent import build_agent_executor
from app.schemas.conversation import (
    ConversationCreate, ConversationUpdate, MemoryItemUpdate, MessageCreate,
)
from app.services.conversation import ConversationService
from app.services.query_resolver import QueryResolver
from app.services.run_manager import RunManager
from app.services.long_term_memory import LongTermMemoryService
from app.observability import (
    bind_context,
    configure_logging,
    configure_tracing,
    llmops,
    mark_span_error,
    new_request_id,
    record_http_request,
    reset_context,
    span,
)

repository = PaperRepository(settings.database_path)
parser = PaperParser(settings.parser_backend, settings.models_dir / "docling")
retriever = HybridRetriever(settings)
reading = ReadingService(settings, retriever)
query_router = QueryRouter()
figure_understanding = FigureUnderstandingService(settings, repository)
agent_executor = build_agent_executor(settings, retriever, reading, figure_understanding)
conversation_service = ConversationService(repository)
long_term_memory = LongTermMemoryService(repository, settings)
query_resolver = QueryResolver(
    reading,
    enabled=settings.query_resolver_enabled,
    recent_turns=settings.conversation_recent_turns,
    max_chars=settings.conversation_context_max_chars,
)
run_manager = RunManager(
    buffer_size=settings.run_event_buffer_size,
    retention_seconds=settings.run_event_retention_seconds,
    heartbeat_seconds=settings.sse_heartbeat_seconds,
)
logger = logging.getLogger("paperlens")
_run_event_callback: ContextVar = ContextVar("paperlens_run_event_callback", default=None)
configure_logging(enabled=settings.structured_logging_enabled, level=settings.log_level)
configure_tracing(
    enabled=settings.llmops_enabled and settings.otel_enabled,
    service_name=settings.otel_service_name,
    endpoint=settings.otel_exporter_endpoint,
)


def _memory_service() -> LongTermMemoryService:
    """Follow a repository replacement used by tests or an embedded deployment."""
    if long_term_memory.repository is repository:
        return long_term_memory
    return LongTermMemoryService(repository, settings)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用退出时释放本地 Qdrant 文件锁。"""
    yield
    retriever.close()


app = FastAPI(title="PaperLens", version="6.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


@app.middleware("http")
async def observe_http_request(request: Request, call_next):
    """Correlate logs/traces without exporting paper or conversation IDs as metric labels."""
    request_id = new_request_id(request.headers.get("X-Request-ID"))
    tokens = bind_context(request_id=request_id)
    started = time.perf_counter()
    status = 500
    try:
        with span("http.request", http_method=request.method):
            response = await call_next(request)
            status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        route_object = request.scope.get("route")
        route = getattr(route_object, "path", "unmatched")
        record_http_request(
            method=request.method,
            route=route,
            status=status,
            duration_seconds=time.perf_counter() - started,
        )
        reset_context(tokens)


class AskRequest(BaseModel):
    """问答请求。"""
    question: str = Field(min_length=3, max_length=500)


class EvaluationRequest(BaseModel):
    """离线评测记录；可同时提交三种 mode。"""
    records: list[dict]
    top_k: int = Field(default=5, ge=1, le=100)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """返回单页应用。"""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    """公开运行状态，不泄露密钥内容。"""
    return {
        "service": "ok",
        "deepseek_key_present": bool(settings.deepseek_key),
        "qwen_vl_key_present": bool(settings.qwen_vl_key),
        "qwen_vl_model": settings.qwen_vl_model,
        "online_visual_judge_enabled": settings.online_visual_judge_enabled,
        "parser_backend": settings.parser_backend,
        "vector_enabled": settings.vector_enabled,
        "reranker_enabled": settings.reranker_enabled,
        "retrieval": retriever.status(),
        "models_dir": str(settings.models_dir),
        "models": model_status(),
    }


@app.get("/api/ready")
def ready() -> dict:
    """Readiness uses local dependencies only and never spends Provider tokens."""
    checks = {
        "database": repository.check_ready(),
        "data_directory": settings.data_dir.is_dir() and settings.data_dir.exists(),
        "checkpoint_directory": settings.langgraph_checkpoint_path.parent.is_dir(),
    }
    ready_state = all(bool(value) for value in checks.values())
    if not ready_state:
        raise HTTPException(503, {"ready": False, "checks": checks})
    return {"ready": True, "checks": checks}


@app.get("/metrics", include_in_schema=False)
def metrics() -> PlainTextResponse:
    if not settings.metrics_enabled:
        raise HTTPException(404, "Metrics 未启用。")
    return PlainTextResponse(
        llmops.render().decode("utf-8"),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.post("/api/evaluation")
def evaluation(request: EvaluationRequest) -> dict:
    """比较 Standard、Text Agentic 与 Multimodal Agentic RAG。"""
    return evaluate_records(request.records, request.top_k)


@app.post("/api/papers")
async def upload(file: UploadFile = File(...)) -> dict:
    """校验、保存、解析论文并持久化精读结果。"""
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "请上传 PDF 文件。")
    staged_path, content_sha256 = await stage_pdf(file)
    paper_id = str(uuid.uuid4())
    paper_dir = settings.upload_dir / paper_id
    pdf_path = paper_dir / "source.pdf"
    filename = Path(file.filename).name
    try:
        existing_id = repository.reserve_upload(paper_id, filename, pdf_path, content_sha256)
    except Exception:
        staged_path.unlink(missing_ok=True)
        raise
    if existing_id:
        staged_path.unlink(missing_ok=True)
        existing = repository.get(existing_id)
        if existing and existing[0].status == "completed":
            return serialize(*existing, reused=True)
        return JSONResponse(
            status_code=202,
            content={
                "id": existing_id,
                "filename": filename,
                "status": repository.get_status(existing_id) or "processing",
                "reused": True,
            },
        )
    try:
        paper_dir.mkdir(parents=True)
        staged_path.replace(pdf_path)
        paper, card, indexed = await run_in_threadpool(
            process_paper,
            paper_id,
            filename,
            pdf_path,
            paper_dir,
            content_sha256,
            1,
        )
        return serialize(paper, card, indexed, reused=False)
    except Exception as error:
        staged_path.unlink(missing_ok=True)
        retriever.delete_index(paper_id)
        repository.release_upload(paper_id, content_sha256)
        shutil.rmtree(paper_dir, ignore_errors=True)
        logger.exception("论文处理失败 paper_id=%s", paper_id)
        raise HTTPException(422, f"论文处理失败：{type(error).__name__}") from error


@app.post("/api/papers/{paper_id}/reanalyze")
def reanalyze(paper_id: str) -> dict:
    """在不复制原始 PDF 的前提下重新解析、索引并生成新版本卡片。"""
    result = repository.get(paper_id)
    if not result:
        raise HTTPException(404, "未找到论文。")
    old_paper, _old_card = result
    try:
        version = repository.begin_reanalysis(paper_id)
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error
    work_dir = settings.incoming_dir / f"reanalyze-{paper_id}-{uuid.uuid4().hex}"
    work_dir.mkdir(parents=True)
    try:
        paper = parser.parse(
            paper_id,
            old_paper.filename,
            Path(old_paper.pdf_path),
            work_dir,
        )
        paper.content_sha256 = old_paper.content_sha256
        paper.analysis_version = version
        _move_reanalysis_figures(paper, Path(old_paper.pdf_path).parent, version)
        figure_understanding.enrich_paper(paper)
        indexed = retriever.index(paper)
        paper.vector_indexed = indexed
        paper.index_version = INDEX_VERSION if indexed else 0
        paper.index_error = "" if indexed else (retriever.last_error or "向量索引未启用")
        card = reading.create_card(paper)
        repository.save(paper, card)
        _memory_service().invalidate_for_version(paper_id, version)
        return serialize(paper, card, indexed, reused=False)
    except Exception as error:
        shutil.rmtree(Path(old_paper.pdf_path).parent / f"figures-v{version}", ignore_errors=True)
        repository.finish_reanalysis_failure(paper_id, f"{type(error).__name__}: {error}")
        logger.exception("论文重新分析失败 paper_id=%s", paper_id)
        raise HTTPException(422, f"论文重新分析失败：{type(error).__name__}") from error
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str) -> dict:
    """读取持久化论文。"""
    result = repository.get(paper_id)
    if not result: raise HTTPException(404, "未找到论文。")
    return serialize(*result)


@app.get("/api/papers/{paper_id}/learning-memory")
def get_paper_learning_memory(paper_id: str) -> dict:
    """Return paper-scoped, cross-conversation learning progress for the local user."""
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    if settings.memory_backfill_enabled:
        _memory_service().backfill(paper_id)
    paper = repository.get(paper_id)[0]
    service = _memory_service()
    service.invalidate_for_version(paper_id, paper.analysis_version)
    service.apply_retention()
    return service.refresh_aggregate(paper_id)


@app.delete("/api/papers/{paper_id}/learning-memory", status_code=204)
def clear_paper_learning_memory(paper_id: str):
    """Clear derived learning memory without deleting conversations or paper evidence."""
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    repository.clear_paper_learning_memory(paper_id)
    return None


@app.get("/api/papers/{paper_id}/learning-memory/items")
def list_paper_memory_items(paper_id: str, include_inactive: bool = False) -> dict:
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    return {"items": _memory_service().list_items(
        paper_id, include_inactive=include_inactive
    )}


@app.patch("/api/papers/{paper_id}/learning-memory/items/{memory_id}")
def update_paper_memory_item(
    paper_id: str, memory_id: str, request: MemoryItemUpdate
) -> dict:
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    value = _memory_service().update_item(
        paper_id, memory_id, **request.model_dump(exclude_none=True)
    )
    if not value:
        raise HTTPException(404, "未找到长期记忆。")
    return value


@app.post("/api/papers/{paper_id}/learning-memory/backfill")
def backfill_paper_memory(paper_id: str) -> dict:
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    return _memory_service().backfill(paper_id)


@app.post("/api/papers/{paper_id}/ask")
def ask(paper_id: str, request: AskRequest) -> dict:
    """检索原文证据，生成回答并持久化历史。"""
    return _ask_paper(
        paper_id, request, event_callback=_run_event_callback.get()
    )


def _ask_paper(paper_id: str, request: AskRequest, *, event_callback=None) -> dict:
    """Internal RAG entry that can publish live orchestration events."""
    def emit(event: str, data: dict) -> None:
        llmops.record_event(event, data)
        if event_callback:
            event_callback(event, data)

    result = repository.get(paper_id)
    if not result: raise HTTPException(404, "未找到论文。")
    paper, _ = result
    if paper.status != "completed":
        raise HTTPException(409, "论文仍在处理中，请稍后重试。")
    started = time.perf_counter()
    initial_tokens = getattr(reading, "token_usage", 0)
    emit("router.started", {"question_length": len(request.question)})
    route = query_router.route(request.question)
    emit("router.completed", {
        "route": route.get("route"),
        "modalities": route.get("modalities", []),
        "required_modalities": route.get("required_modalities", []),
    })
    if route["route"] == "agentic_rag":
        with span("rag.agentic", route="agentic_rag", modalities=route.get("modalities", [])):
            result = agent_executor.run(
                paper, request.question, route, event_callback=emit
            )
        repository.save_trace(paper_id, result["trace"])
        repository.save_answer(
            paper_id,
            request.question,
            result["answer"],
            result.get("citations", []),
            answerable=result.get("answerable"),
            claims=result.get("claims", []),
            query_plan={"router": route, "plan": result.get("plan", {})},
            retrieval={
                **result.get("retrieval", {}),
                "sufficiency": result.get("sufficiency", {}),
                "trace": result.get("trace", {}),
            },
            latency_ms=result.get("latency_ms"),
            model_name=settings.deepseek_model,
            status=result.get("status", "ok"),
        )
        _record_rag_outcome(result)
        return result
    query_plan = reading.plan_query(request.question)
    emit("tool.started", {
        "agent": "standard_rag", "task_id": "standard_rag",
        "tool": "search_text", "query_length": len(query_plan["semantic_query"]),
        "protocol": "fixed",
    })
    results, retrieval_trace = retriever.search_with_trace(
        paper.chunks,
        query_plan["semantic_query"],
        lexical_query=query_plan["lexical_query"],
        section_hints=tuple(query_plan.get("section_hints", ())),
    )
    if settings.vector_enabled and (
        not paper.vector_indexed or paper.index_version != INDEX_VERSION
    ):
        retrieval_trace["degraded"] = True
        retrieval_trace["warnings"].append(
            paper.index_error or "该论文没有可确认的当前版本多模态向量索引，请重新分析或重建索引"
        )
    retrieval_trace["paper_index_version"] = paper.index_version
    chunks = [item.chunk for item in results]
    grounded = reading.answer_with_evidence(request.question, chunks, paper.sentences)
    citations = grounded["citations"]
    if not settings.deepseek_key:
        citations = [citation_payload(item.chunk, item.score) for item in results]
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    trace = {
        "run_id": str(uuid.uuid4()),
        "query": request.question,
        "route": "standard_rag",
        "router": route,
        "plan": {},
        "rewrites": [query_plan],
        "steps": [{
            "step": 1,
            "subtask_id": "standard_rag",
            "tool": "search_text",
            "arguments": {"query": query_plan["semantic_query"]},
            "result_ids": [item.get("evidence_id", item.get("id")) for item in citations],
            "latency_ms": latency_ms,
            "cached": False,
            "retrieval": retrieval_trace,
        }],
        "final_evidence_ids": [item.get("evidence_id", item.get("id")) for item in citations],
        "total_steps": 1,
        "total_latency_ms": latency_ms,
        "token_usage": getattr(reading, "token_usage", 0) - initial_tokens,
        "qwen_vl_calls": 0,
        "sufficiency": {
            "sufficient": grounded.get("answerable") is True,
            "covered_subtasks": ["standard_rag"] if grounded.get("answerable") is True else [],
            "missing_subtasks": [] if grounded.get("answerable") is True else ["standard_rag"],
            "missing_information": [],
        },
        "evidence_memory": [],
        "retrieval": retrieval_trace,
    }
    emit("tool.completed", {
        "agent": "standard_rag", "task_id": "standard_rag",
        "tool": "search_text",
        "result_count": len(trace["final_evidence_ids"]),
        "latency_ms": latency_ms,
        "status": "success" if citations else "partial",
        "protocol": "fixed",
    })
    emit("answer.verified", {
        "node": "standard_grounded_answer",
        "status": grounded.get("status", "unknown"),
        "answerable": grounded.get("answerable"),
        "citation_count": len(citations),
    })
    repository.save_trace(paper_id, trace)
    repository.save_answer(
        paper_id,
        request.question,
        grounded["answer"],
        citations,
        answerable=grounded.get("answerable"),
        claims=grounded.get("claims", []),
        query_plan=query_plan,
        retrieval=retrieval_trace,
        latency_ms=latency_ms,
        model_name=settings.deepseek_model,
        status=grounded.get("status", "ok"),
    )
    response = {
        **grounded,
        "route": "standard_rag",
        "mode": "standard_rag",
        "evidence_sufficient": grounded.get("answerable") is True,
        "sufficiency": trace["sufficiency"],
        "router": route,
        "trace": trace,
        "citations": citations,
        "query_plan": query_plan,
        "retrieval": retrieval_trace,
        "latency_ms": latency_ms,
    }
    _record_rag_outcome(response)
    return response


def _record_rag_outcome(result: dict) -> None:
    route = str(result.get("route", "unknown"))
    status = str(result.get("status", "unknown"))
    llmops.runs.labels(route=route, status=status).inc()
    latency_ms = result.get("latency_ms")
    if isinstance(latency_ms, (int, float)):
        llmops.run_duration.labels(route=route).observe(max(0.0, latency_ms / 1000.0))
    llmops.answer_outcomes.labels(
        answerable=str(result.get("answerable")).lower(),
        evidence_sufficient=str(result.get("evidence_sufficient")).lower(),
        status=status,
    ).inc()
    logger.info("rag_run_completed", extra={"fields": {
        "route": route,
        "status": status,
        "latency_ms": latency_ms,
        "answerable": result.get("answerable"),
        "evidence_sufficient": result.get("evidence_sufficient"),
        "citation_count": len(result.get("citations", [])),
        "token_usage": result.get("trace", {}).get("token_usage", 0),
        "qwen_vl_calls": result.get("trace", {}).get("qwen_vl_calls", 0),
    }})


@app.post("/api/papers/{paper_id}/conversations", status_code=201)
def create_conversation(paper_id: str, request: ConversationCreate) -> dict:
    """Create a persistent conversation scoped to one completed paper."""
    if not settings.conversations_enabled:
        raise HTTPException(404, "Conversation 功能未启用。")
    try:
        return conversation_service.create(paper_id, request.title)
    except KeyError as error:
        raise HTTPException(404, "未找到论文。") from error
    except RuntimeError as error:
        raise HTTPException(409, "论文仍在处理中，请稍后重试。") from error


@app.get("/api/papers/{paper_id}/conversations")
def list_conversations(paper_id: str, include_archived: bool = False) -> dict:
    if not repository.get(paper_id):
        raise HTTPException(404, "未找到论文。")
    return {"items": repository.list_conversations(paper_id, include_archived=include_archived)}


@app.get("/api/papers/{paper_id}/conversations/{conversation_id}")
def get_conversation(paper_id: str, conversation_id: str) -> dict:
    value = repository.get_conversation(paper_id, conversation_id)
    if not value:
        raise HTTPException(404, "未找到对话。")
    return value


@app.patch("/api/papers/{paper_id}/conversations/{conversation_id}")
def rename_conversation(
    paper_id: str, conversation_id: str, request: ConversationUpdate
) -> dict:
    value = repository.update_conversation(
        paper_id, conversation_id, title=request.title
    )
    if not value:
        raise HTTPException(404, "未找到对话。")
    return value


@app.delete("/api/papers/{paper_id}/conversations/{conversation_id}", status_code=204)
def archive_conversation(paper_id: str, conversation_id: str):
    try:
        archived = repository.archive_conversation(paper_id, conversation_id)
    except RuntimeError as error:
        raise HTTPException(409, "对话仍有运行中的任务，请先停止。") from error
    if not archived:
        raise HTTPException(404, "未找到对话。")
    return None


@app.post(
    "/api/papers/{paper_id}/conversations/{conversation_id}/messages",
    status_code=202,
)
async def submit_conversation_message(
    paper_id: str, conversation_id: str, request: MessageCreate
) -> dict:
    """Persist the user turn, return immediately and execute the RAG run in background."""
    try:
        message, run, created = conversation_service.submit(
            paper_id, conversation_id, request.question, request.client_message_id
        )
    except KeyError as error:
        raise HTTPException(404, "未找到对话，或对话不属于当前论文。") from error
    except RuntimeError as error:
        reasons = {
            "conversation_archived": "对话已归档。",
            "active_run": "当前对话已有运行中的任务。",
        }
        raise HTTPException(409, reasons.get(str(error), "当前无法提交消息。")) from error
    if created:
        run_manager.start(
            run["id"],
            lambda: _execute_conversation_run(paper_id, conversation_id, run["id"]),
        )
    return {
        "conversation_id": conversation_id,
        "message_id": message["id"],
        "run_id": run["id"],
        "status": run["status"],
        "created": created,
    }


@app.get("/api/papers/{paper_id}/conversations/{conversation_id}/runs/{run_id}")
def get_conversation_run(paper_id: str, conversation_id: str, run_id: str) -> dict:
    value = repository.get_run(paper_id, conversation_id, run_id)
    if not value:
        raise HTTPException(404, "未找到运行记录。")
    return value


@app.get("/api/papers/{paper_id}/conversations/{conversation_id}/runs/{run_id}/events")
async def stream_conversation_run(
    paper_id: str,
    conversation_id: str,
    run_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    run = repository.get_run(paper_id, conversation_id, run_id)
    if not run:
        raise HTTPException(404, "未找到运行记录。")
    if not settings.sse_enabled:
        raise HTTPException(409, "SSE 功能未启用。")
    terminal = run["status"] in {
        "completed", "partial", "failed", "cancelled", "timed_out", "interrupted"
    }
    run_manager.register(run_id)
    if terminal:
        await run_manager.publish(
            run_id,
            f"run.{run['status'] if run['status'] != 'interrupted' else 'failed'}",
            {"run_id": run_id, "status": run["status"], "recovered": True},
        )
    try:
        cursor = max(0, int(last_event_id or "0"))
    except ValueError:
        cursor = 0
    return StreamingResponse(
        run_manager.events(run_id, cursor),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/papers/{paper_id}/conversations/{conversation_id}/runs/{run_id}/cancel")
async def cancel_conversation_run(paper_id: str, conversation_id: str, run_id: str) -> dict:
    run = repository.get_run(paper_id, conversation_id, run_id)
    if not run:
        raise HTTPException(404, "未找到运行记录。")
    accepted = repository.request_run_cancel(paper_id, conversation_id, run_id)
    if accepted:
        await run_manager.publish(run_id, "run.cancel_requested", {"run_id": run_id})
    return {"run_id": run_id, "cancel_requested": accepted, "status": run["status"]}


async def _execute_conversation_run(
    paper_id: str, conversation_id: str, run_id: str
) -> None:
    """Execute one isolated turn; database writes happen only at short boundaries."""
    run = repository.get_run(paper_id, conversation_id, run_id)
    if not run or not repository.mark_run_running(paper_id, conversation_id, run_id):
        return
    context_tokens = bind_context(
        run_id=run_id, conversation_id=conversation_id, paper_id=paper_id
    )
    llmops.active_runs.inc()
    question_preview = re.sub(
        r"\s+", " ", str(run.get("original_question", "")).strip()
    )[:240]
    run_span = span(
        "paperlens.conversation_run",
        run_type="conversation",
        run_id=run_id,
        conversation_id=conversation_id,
        paper_id=paper_id,
        question_preview=question_preview,
        question_length=len(str(run.get("original_question", ""))),
    )
    current_span = run_span.__enter__()
    await run_manager.publish(run_id, "run.started", {"run_id": run_id})
    try:
        conversation = repository.get_conversation(paper_id, conversation_id)
        if not conversation:
            raise RuntimeError("conversation disappeared")
        if settings.memory_backfill_enabled:
            _memory_service().backfill(paper_id)
        conversation["paper_memory"] = _memory_service().refresh_aggregate(paper_id)
        await run_manager.publish(run_id, "context.loading", {"run_id": run_id})
        resolution = await asyncio.to_thread(
            query_resolver.resolve, run["original_question"], conversation
        )
        resolved = resolution["standalone_question"]
        await run_manager.publish(run_id, "query.resolved", {
            "run_id": run_id,
            "original_question": run["original_question"],
            "resolved_question": resolved,
            "resolver_used": resolution.get("resolver_used", False),
            "needs_clarification": resolution.get("needs_clarification", False),
        })
        if resolution.get("intent") == "conversation_history":
            answer, history = query_resolver.answer_conversation_history(
                conversation, current_message_id=run["user_message_id"]
            )
            result_ids = [f"turn-{value}" for value in history["turn_indexes"]]
            trace = {
                "route": "conversation_memory",
                "mode": "conversation_memory",
                "steps": [{
                    "step": 1,
                    "tool": "read_conversation_history",
                    "result_ids": result_ids,
                    "latency_ms": 0.0,
                }],
                "conversation_id": conversation_id,
                "original_question": run["original_question"],
                "resolved_question": resolved,
                "context_message_count": resolution.get("context_message_count", 0),
                "referenced_evidence_ids": [],
                "stream_event_count": 0,
            }
            result = {
                "answer": answer,
                "status": "ok",
                "answerable": True,
                "evidence_sufficient": True,
                "citations": [],
                "route": "conversation_memory",
                "mode": "conversation_memory",
                "trace": trace,
                "history": history,
            }
            await run_manager.publish(run_id, "memory.completed", history)
            await _publish_answer(run_id, answer)
            repository.finish_conversation_run(
                paper_id=paper_id,
                conversation_id=conversation_id,
                run_id=run_id,
                status="completed",
                answer=answer,
                resolved_question=resolved,
                citations=[],
                metadata={
                    "trace": trace,
                    "resolution": resolution,
                    "answerable": True,
                    "evidence_sufficient": True,
                    "response_source": "conversation_history",
                },
                result=result,
            )
            _update_conversation_memory(paper_id, conversation_id)
            await run_manager.publish(run_id, "run.completed", {
                "run_id": run_id,
                "status": "completed",
                "citations": [],
                "trace": trace,
            })
            return
        if resolution.get("intent") == "paper_learning_memory":
            answer, details = query_resolver.answer_paper_learning_memory(
                conversation["paper_memory"]
            )
            trace = {
                "route": "paper_learning_memory",
                "mode": "paper_learning_memory",
                "steps": [{
                    "step": 1,
                    "tool": "read_paper_learning_memory",
                    "result_ids": details.get("result_ids", []),
                    "latency_ms": 0.0,
                }],
                "conversation_id": conversation_id,
                "original_question": run["original_question"],
                "resolved_question": resolved,
                "context_message_count": resolution.get("context_message_count", 0),
                "referenced_evidence_ids": [],
                "stream_event_count": 0,
            }
            result = {
                "answer": answer,
                "status": "ok",
                "answerable": True,
                "evidence_sufficient": True,
                "citations": [],
                "route": "paper_learning_memory",
                "mode": "paper_learning_memory",
                "trace": trace,
                "learning_memory": details,
            }
            await run_manager.publish(run_id, "memory.completed", details)
            await _publish_answer(run_id, answer)
            repository.finish_conversation_run(
                paper_id=paper_id,
                conversation_id=conversation_id,
                run_id=run_id,
                status="completed",
                answer=answer,
                resolved_question=resolved,
                citations=[],
                metadata={
                    "trace": trace,
                    "resolution": resolution,
                    "answerable": True,
                    "evidence_sufficient": True,
                    "response_source": "paper_learning_memory",
                },
                result=result,
            )
            _update_conversation_memory(paper_id, conversation_id)
            await run_manager.publish(run_id, "run.completed", {
                "run_id": run_id,
                "status": "completed",
                "citations": [],
                "trace": trace,
            })
            return
        if resolution.get("needs_clarification"):
            answer = resolution.get("clarification_question") or "请补充你所指的对象。"
            repository.finish_conversation_run(
                paper_id=paper_id, conversation_id=conversation_id, run_id=run_id,
                status="partial", answer=answer, resolved_question=resolved,
                result={"resolution": resolution, "status": "needs_clarification"},
                metadata={"resolution": resolution},
            )
            await _publish_answer(run_id, answer)
            await run_manager.publish(run_id, "run.partial", {
                "run_id": run_id, "status": "partial", "needs_clarification": True
            })
            return
        if repository.is_cancel_requested(paper_id, conversation_id, run_id):
            await _finish_cancelled(paper_id, conversation_id, run_id, resolved)
            return
        event_loop = asyncio.get_running_loop()

        def publish_live_event(event: str, data: dict) -> None:
            future = asyncio.run_coroutine_threadsafe(
                run_manager.publish(run_id, event, data), event_loop
            )
            future.result(timeout=5)

        callback_token = _run_event_callback.set(publish_live_event)
        try:
            result = await asyncio.to_thread(ask, paper_id, AskRequest(question=resolved))
        finally:
            _run_event_callback.reset(callback_token)
        trace = result.get("trace", {})
        if repository.is_cancel_requested(paper_id, conversation_id, run_id):
            await _finish_cancelled(paper_id, conversation_id, run_id, resolved)
            return
        answer = str(result.get("answer", ""))
        await _publish_answer(run_id, answer)
        status = "completed" if result.get("status", "ok") == "ok" else "partial"
        trace.update({
            "conversation_id": conversation_id,
            "turn_index": next(
                (m["turn_index"] for m in conversation["messages"] if m["id"] == run["user_message_id"]),
                None,
            ),
            "original_question": run["original_question"],
            "resolved_question": resolved,
            "context_message_count": resolution.get("context_message_count", 0),
            "referenced_evidence_ids": resolution.get("referenced_evidence_ids", []),
            "stream_event_count": 0,
        })
        citations = result.get("citations", [])
        repository.finish_conversation_run(
            paper_id=paper_id, conversation_id=conversation_id, run_id=run_id,
            status=status, answer=answer, resolved_question=resolved,
            citations=citations, metadata={
                "trace": trace,
                "resolution": resolution,
                "answerable": result.get("answerable"),
                "evidence_sufficient": result.get("evidence_sufficient"),
                "visual_verification_status": result.get("visual_verification_status"),
                "visual_answer_check": result.get("visual_answer_check", {}),
                "insufficient_evidence": result.get("insufficient_evidence", []),
            },
            result=result,
        )
        _update_paper_learning_memory(
            paper_id=paper_id,
            conversation_id=conversation_id,
            run=run,
            resolved_question=resolved,
            status=status,
            result=result,
        )
        _update_conversation_memory(paper_id, conversation_id)
        await run_manager.publish(run_id, f"run.{status}", {
            "run_id": run_id, "status": status,
            "citations": citations, "trace": trace,
        })
    except asyncio.CancelledError:
        await _finish_cancelled(paper_id, conversation_id, run_id, "")
        raise
    except Exception as error:
        mark_span_error(current_span, error)
        logger.exception("conversation run failed run_id=%s", run_id)
        repository.finish_conversation_run(
            paper_id=paper_id, conversation_id=conversation_id, run_id=run_id,
            status="failed", answer="问答执行失败，请稍后重试。", error={
                "code": type(error).__name__.upper(),
                "message": "问答执行失败，请稍后重试。",
            },
        )
        await run_manager.publish(run_id, "run.failed", {
            "run_id": run_id, "status": "failed", "error": "问答执行失败，请稍后重试。"
        })
    finally:
        final_run = repository.get_run(paper_id, conversation_id, run_id)
        logger.info("conversation_run_finished", extra={"fields": {
            "status": final_run.get("status") if final_run else "unknown",
            "original_question_length": len(run.get("original_question", "")),
        }})
        run_span.__exit__(None, None, None)
        llmops.active_runs.dec()
        reset_context(context_tokens)


async def _publish_answer(run_id: str, answer: str) -> None:
    """Stream the verified final text; unverified model tokens are never exposed."""
    await run_manager.publish(run_id, "answer.started", {"run_id": run_id})
    size = settings.sse_answer_chunk_chars
    pieces = [answer[start:start + size] for start in range(0, len(answer), size)]
    for index, piece in enumerate(pieces):
        await run_manager.publish(run_id, "answer.delta", {"delta": piece})
        # A zero-delay yield caused browsers to paint all buffered events in a
        # single frame. A small configurable delay makes transport streaming
        # observable while retaining the verify-before-display contract.
        if index < len(pieces) - 1:
            await asyncio.sleep(settings.sse_answer_chunk_delay_seconds)
    await run_manager.publish(run_id, "answer.completed", {"run_id": run_id})


async def _finish_cancelled(
    paper_id: str, conversation_id: str, run_id: str, resolved_question: str
) -> None:
    repository.finish_conversation_run(
        paper_id=paper_id, conversation_id=conversation_id, run_id=run_id,
        status="cancelled", answer="本轮回答已取消。", resolved_question=resolved_question,
        error={"code": "CANCELLED", "message": "用户已取消运行。"},
    )
    await run_manager.publish(run_id, "run.cancelled", {
        "run_id": run_id, "status": "cancelled"
    })


def _update_conversation_memory(paper_id: str, conversation_id: str) -> None:
    """Maintain bounded navigational memory; it is never used as citation evidence."""
    conversation = repository.get_conversation(paper_id, conversation_id)
    if not conversation:
        return
    user_messages = [m for m in conversation["messages"] if m["role"] == "user"]
    citations = [
        str(c.get("evidence_id", c.get("id", "")))
        for m in conversation["messages"][-12:]
        for c in m.get("citations", [])
        if c.get("evidence_id", c.get("id"))
    ]
    memory = {
        **conversation.get("memory", {}),
        "referenced_evidence_ids": list(dict.fromkeys(citations))[-20:],
        "summary_version": 1,
    }
    updates: dict = {"memory": memory}
    if len(user_messages) >= settings.conversation_summary_trigger_turns:
        snippets = [m["content"].strip() for m in conversation["messages"][-12:]]
        updates["summary"] = " / ".join(snippets)[-4000:]
    if conversation["title"] == "新对话" and user_messages:
        updates["title"] = user_messages[0]["content"].strip()[:32]
    repository.update_conversation(paper_id, conversation_id, **updates)


def _update_paper_learning_memory(
    *,
    paper_id: str,
    conversation_id: str,
    run: dict,
    resolved_question: str,
    status: str,
    result: dict,
) -> None:
    """Delegate versioning, scoring, expiration and aggregation to the memory service."""
    _memory_service().record_run(
        paper_id=paper_id,
        conversation_id=conversation_id,
        run=run,
        resolved_question=resolved_question,
        status=status,
        result=result,
    )


@app.get("/api/papers/{paper_id}/evidence/{evidence_id}")
def evidence(paper_id: str, evidence_id: str) -> dict:
    """按统一 evidence_id 读取文本、Figure 或 Table 原始证据。"""
    result = repository.get(paper_id)
    if not result:
        raise HTTPException(404, "未找到论文。")
    paper, _ = result
    sentence = next((item for item in paper.sentences if item.id == evidence_id), None)
    if sentence:
        return sentence_evidence(sentence).to_payload()
    chunk = next((item for item in paper.chunks if item.id == evidence_id), None)
    if chunk:
        return citation_payload(chunk)
    figure_item = next((item for item in paper.figures if item.id == evidence_id), None)
    if figure_item:
        return figure_item.as_evidence().to_payload()
    raise HTTPException(404, "未找到 Evidence。")


@app.get("/api/papers/{paper_id}/traces/{run_id}")
def trace(paper_id: str, run_id: str) -> dict:
    """返回可审计的 Router/Planner/Tool/Evidence 执行轨迹。"""
    value = repository.get_trace(paper_id, run_id)
    if not value:
        raise HTTPException(404, "未找到 Agent Trace。")
    return value


@app.get("/api/papers/{paper_id}/pdf")
def pdf(paper_id: str) -> FileResponse:
    """以内联方式返回原 PDF。"""
    result = repository.get(paper_id)
    if not result: raise HTTPException(404, "未找到论文。")
    paper, _ = result
    return FileResponse(paper.pdf_path, media_type="application/pdf", headers={"Content-Disposition":"inline"})


@app.get("/api/papers/{paper_id}/figures/{figure_id}")
def figure(paper_id: str, figure_id: str) -> FileResponse:
    """安全返回已持久化的位图，供前端展示实验图和论文插图。"""
    result = repository.get(paper_id)
    if not result:
        raise HTTPException(404, "未找到论文。")
    item = next((value for value in result[0].figures if value.id == figure_id), None)
    if not item or not item.image_path or not Path(item.image_path).is_file():
        raise HTTPException(404, "未找到可直接展示的位图；该对象可能是矢量图或表格。")
    return FileResponse(item.image_path)


def serialize(paper, card, indexed: bool | None = None, reused: bool = False) -> dict:
    """统一论文 API 输出。"""
    index_ready = paper.vector_indexed if indexed is None else indexed
    return {"id":paper.id, "filename":paper.filename, "title":paper.title, "abstract":paper.abstract, "parser":paper.parser_name, "status":paper.status, "analysis_version":paper.analysis_version, "reused":reused, "chunk_count":len(paper.chunks), "vector_indexed":index_ready, "index_version":paper.index_version, "index_error":paper.index_error, "pdf_url":f"/api/papers/{paper.id}/pdf", "card":card, "figures":[{"id":f.id,"evidence_id":f.id,"type":f.evidence_type,"page":f.page,"kind":f.kind,"section":f.section,"caption":f.caption,"bbox":f.bbox,"has_table_content":bool(f.content),"visual_status":f.description.get("status", "not_applicable" if f.kind == "table" else "pending"),"image_url":f"/api/papers/{paper.id}/figures/{f.id}" if f.image_path else None} for f in paper.figures]}


async def stage_pdf(file: UploadFile) -> tuple[Path, str]:
    """流式保存并校验 PDF，返回临时路径和内容哈希。"""
    path = settings.incoming_dir / f"{uuid.uuid4().hex}.pdf"
    digest = hashlib.sha256()
    size = 0
    header = bytearray()
    try:
        with path.open("wb") as stream:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise HTTPException(413, f"文件超过 {settings.max_upload_bytes // 1024 // 1024}MB 限制。")
                if len(header) < 1024:
                    header.extend(block[:1024 - len(header)])
                digest.update(block)
                stream.write(block)
        if size == 0:
            raise HTTPException(400, "PDF 文件为空。")
        if b"%PDF-" not in bytes(header):
            raise HTTPException(400, "文件内容不是有效的 PDF。")
        return path, digest.hexdigest()
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _move_reanalysis_figures(paper, paper_dir: Path, version: int) -> None:
    """将重新分析生成的图片移入版本目录并更新领域对象路径。"""
    destination = paper_dir / f"figures-v{version}"
    shutil.rmtree(destination, ignore_errors=True)
    for figure in paper.figures:
        if not figure.image_path:
            continue
        source = Path(figure.image_path)
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        shutil.move(str(source), str(target))
        figure.image_path = str(target)


def process_paper(
    paper_id: str,
    filename: str,
    pdf_path: Path,
    output_dir: Path,
    content_sha256: str,
    analysis_version: int,
):
    """在线程池中执行 CPU/阻塞 I/O 密集的解析、索引与模型调用。"""
    paper = parser.parse(paper_id, filename, pdf_path, output_dir)
    paper.content_sha256 = content_sha256
    paper.analysis_version = analysis_version
    figure_understanding.enrich_paper(paper)
    indexed = retriever.index(paper)
    paper.vector_indexed = indexed
    paper.index_version = INDEX_VERSION if indexed else 0
    paper.index_error = "" if indexed else (retriever.last_error or "向量索引未启用")
    card = reading.create_card(paper)
    repository.save(paper, card)
    return paper, card, indexed


def model_status() -> dict:
    """只检查本地权重和不完整下载标记，不触发任何网络下载。"""
    root = settings.models_dir
    docling_root = root / "docling"
    docling_sets = [
        path for path in docling_root.iterdir() if path.is_dir()
        and any(file.is_file() for file in path.rglob("*") if ".cache" not in file.parts)
    ] if docling_root.exists() else []
    return {
        "bge_m3": (root / "modelscope" / "bge-m3" / "pytorch_model.bin").is_file(),
        "bge_reranker": (root / "modelscope" / "bge-reranker-v2-m3" / "model.safetensors").is_file(),
        "docling_layout": (root / "docling" / "docling-project--docling-layout-heron" / "model.safetensors").is_file(),
        "docling_academic_pdf_set": len(docling_sets) >= 5,
        "docling_model_directories": len(docling_sets),
        "docling_incomplete_files": len(list((root / "docling").rglob("*.incomplete"))) + len(list((root / "docling").rglob("*.lock"))),
    }
