"""PaperLens Web API。"""

from contextlib import asynccontextmanager
import hashlib
import logging
from pathlib import Path
import shutil
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
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

repository = PaperRepository(settings.database_path)
parser = PaperParser(settings.parser_backend, settings.models_dir / "docling")
retriever = HybridRetriever(settings)
reading = ReadingService(settings, retriever)
query_router = QueryRouter()
figure_understanding = FigureUnderstandingService(settings, repository)
agent_executor = build_agent_executor(settings, retriever, reading, figure_understanding)
logger = logging.getLogger("paperlens")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用退出时释放本地 Qdrant 文件锁。"""
    yield
    retriever.close()


app = FastAPI(title="PaperLens", version="3.0", lifespan=lifespan)


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


@app.post("/api/papers/{paper_id}/ask")
def ask(paper_id: str, request: AskRequest) -> dict:
    """检索原文证据，生成回答并持久化历史。"""
    result = repository.get(paper_id)
    if not result: raise HTTPException(404, "未找到论文。")
    paper, _ = result
    if paper.status != "completed":
        raise HTTPException(409, "论文仍在处理中，请稍后重试。")
    started = time.perf_counter()
    initial_tokens = getattr(reading, "token_usage", 0)
    route = query_router.route(request.question)
    if route["route"] == "agentic_rag":
        result = agent_executor.run(paper, request.question, route)
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
        return result
    query_plan = reading.plan_query(request.question)
    results, retrieval_trace = retriever.search_with_trace(
        paper.chunks,
        query_plan["semantic_query"],
        lexical_query=query_plan["lexical_query"],
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
    return {
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
