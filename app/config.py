"""应用配置：所有本地数据、模型与服务开关集中在此处管理。"""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    """运行配置，避免在业务代码中散落环境变量读取。"""

    data_dir: Path = PROJECT_DIR / "data"
    models_dir: Path = PROJECT_DIR / "models"
    max_upload_bytes: int = 30 * 1024 * 1024
    parser_backend: str = os.getenv("PARSER_BACKEND", "docling")
    vector_enabled: bool = os.getenv("ENABLE_VECTOR_SEARCH", "false").lower() == "true"
    reranker_enabled: bool = os.getenv("ENABLE_RERANKER", "false").lower() == "true"
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    deepseek_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    deepseek_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    qwen_vl_key: str = os.getenv("QWEN_VL_API_KEY", "").strip()
    qwen_vl_url: str = os.getenv(
        "QWEN_VL_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    qwen_vl_model: str = os.getenv("QWEN_VL_MODEL", "qwen-vl-max")
    reranker_candidate_limit: int = max(
        5, int(os.getenv("RERANK_CANDIDATE_LIMIT", "8"))
    )
    reranker_max_length: int = max(
        64, int(os.getenv("RERANK_MAX_LENGTH", "256"))
    )
    agent_max_steps: int = max(1, int(os.getenv("AGENT_MAX_STEPS", "6")))
    agent_figure_read_limit: int = max(1, int(os.getenv("AGENT_FIGURE_READ_LIMIT", "2")))
    online_visual_judge_enabled: bool = (
        os.getenv("ENABLE_ONLINE_VISUAL_JUDGE", "false").lower() == "true"
    )
    agent_orchestrator: str = os.getenv("AGENT_ORCHESTRATOR", "langgraph").strip().lower()
    multi_agent_max_steps: int = max(
        1, int(os.getenv("MULTI_AGENT_MAX_STEPS", "10"))
    )
    multi_agent_max_retries: int = max(
        0, int(os.getenv("MULTI_AGENT_MAX_RETRIES", "1"))
    )
    multi_agent_timeout_seconds: float = max(
        1.0, float(os.getenv("MULTI_AGENT_TIMEOUT_SECONDS", "120"))
    )
    multi_agent_parallel_enabled: bool = (
        os.getenv("MULTI_AGENT_PARALLEL_ENABLED", "false").lower() == "true"
    )
    multi_agent_max_model_calls: int = max(
        1, int(os.getenv("MULTI_AGENT_MAX_MODEL_CALLS", "8"))
    )
    multi_agent_max_qwen_vl_calls: int = max(
        0, int(os.getenv("MULTI_AGENT_MAX_QWEN_VL_CALLS", "2"))
    )
    langgraph_checkpoint_enabled: bool = (
        os.getenv("LANGGRAPH_CHECKPOINT_ENABLED", "true").lower() == "true"
    )
    langgraph_checkpoint_path_value: str = os.getenv(
        "LANGGRAPH_CHECKPOINT_PATH", "data/langgraph-checkpoints.sqlite3"
    ).strip()
    table_vlm_fallback_enabled: bool = (
        os.getenv("TABLE_VLM_FALLBACK_ENABLED", "false").lower() == "true"
    )
    remote_max_retries: int = max(0, int(os.getenv("REMOTE_MAX_RETRIES", "2")))
    remote_retry_base_delay_seconds: float = max(
        0.0, float(os.getenv("REMOTE_RETRY_BASE_DELAY_SECONDS", "0.25"))
    )
    conversations_enabled: bool = (
        os.getenv("ENABLE_CONVERSATIONS", "true").lower() == "true"
    )
    query_resolver_enabled: bool = (
        os.getenv("ENABLE_QUERY_RESOLVER", "true").lower() == "true"
    )
    conversation_recent_turns: int = max(
        1, int(os.getenv("CONVERSATION_RECENT_TURNS", "6"))
    )
    conversation_summary_trigger_turns: int = max(
        2, int(os.getenv("CONVERSATION_SUMMARY_TRIGGER_TURNS", "6"))
    )
    conversation_context_max_chars: int = max(
        1000, int(os.getenv("CONVERSATION_CONTEXT_MAX_CHARS", "12000"))
    )
    sse_enabled: bool = os.getenv("ENABLE_SSE", "true").lower() == "true"
    sse_heartbeat_seconds: float = max(
        1.0, float(os.getenv("SSE_HEARTBEAT_SECONDS", "15"))
    )
    sse_answer_chunk_chars: int = max(
        1, int(os.getenv("SSE_ANSWER_CHUNK_CHARS", "12"))
    )
    sse_answer_chunk_delay_seconds: float = max(
        0.0, float(os.getenv("SSE_ANSWER_CHUNK_DELAY_MS", "30")) / 1000.0
    )
    run_event_buffer_size: int = max(
        10, int(os.getenv("RUN_EVENT_BUFFER_SIZE", "200"))
    )
    run_event_retention_seconds: int = max(
        30, int(os.getenv("RUN_EVENT_RETENTION_SECONDS", "600"))
    )
    specialist_execution_mode: str = os.getenv(
        "SPECIALIST_EXECUTION_MODE", "function_calling_with_fallback"
    ).strip().lower()
    function_call_max_steps: int = max(
        1, int(os.getenv("FUNCTION_CALL_MAX_STEPS", "4"))
    )
    function_call_max_model_rounds: int = max(
        1, int(os.getenv("FUNCTION_CALL_MAX_MODEL_ROUNDS", "3"))
    )
    function_call_strict: bool = (
        os.getenv("FUNCTION_CALL_STRICT", "true").lower() == "true"
    )
    function_call_tool_result_max_chars: int = max(
        1000, int(os.getenv("FUNCTION_CALL_TOOL_RESULT_MAX_CHARS", "16000"))
    )
    function_call_circuit_failure_threshold: int = max(
        1, int(os.getenv("FUNCTION_CALL_CIRCUIT_FAILURE_THRESHOLD", "3"))
    )
    function_call_circuit_cooldown_seconds: float = max(
        1.0, float(os.getenv("FUNCTION_CALL_CIRCUIT_COOLDOWN_SECONDS", "60"))
    )
    llmops_enabled: bool = os.getenv("ENABLE_LLMOPS", "true").lower() == "true"
    structured_logging_enabled: bool = (
        os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    metrics_enabled: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
    otel_enabled: bool = os.getenv("ENABLE_OTEL", "true").lower() == "true"
    otel_service_name: str = os.getenv("OTEL_SERVICE_NAME", "paperlens").strip()
    otel_exporter_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    memory_retention_days: int = max(
        1, int(os.getenv("MEMORY_RETENTION_DAYS", "365"))
    )
    memory_max_items_per_paper: int = max(
        10, int(os.getenv("MEMORY_MAX_ITEMS_PER_PAPER", "200"))
    )
    memory_backfill_enabled: bool = (
        os.getenv("MEMORY_BACKFILL_ENABLED", "true").lower() == "true"
    )

    @property
    def upload_dir(self) -> Path:
        """返回原始 PDF 的本地保存目录。"""
        return self.data_dir / "uploads"

    @property
    def incoming_dir(self) -> Path:
        """返回尚未完成校验和入库的上传临时目录。"""
        return self.data_dir / "incoming"

    @property
    def qdrant_dir(self) -> Path:
        """返回本地 Qdrant 的持久化数据目录。"""
        return self.data_dir / "qdrant"

    @property
    def database_path(self) -> Path:
        """返回 SQLite 数据库位置。"""
        return self.data_dir / "paperlens.sqlite3"

    @property
    def langgraph_checkpoint_path(self) -> Path:
        """返回独立于业务库的 LangGraph Checkpoint 路径。"""
        value = Path(self.langgraph_checkpoint_path_value)
        return value if value.is_absolute() else PROJECT_DIR / value


settings = Settings()
for directory in (
    settings.data_dir,
    settings.upload_dir,
    settings.incoming_dir,
    settings.qdrant_dir,
    settings.models_dir,
):
    directory.mkdir(parents=True, exist_ok=True)

# 统一预创建模型缓存目录，便于用户确认所有模型资产都受 demo 管理。
for directory in (
    settings.models_dir / "huggingface",
    settings.models_dir / "docling",
    settings.models_dir / "modelscope",
):
    directory.mkdir(parents=True, exist_ok=True)

# 第三方模型库在首次下载时读取这些变量，避免缓存落入用户主目录。
os.environ.setdefault("HF_HOME", str(settings.models_dir / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(settings.models_dir / "huggingface" / "hub"))
os.environ.setdefault("DOCLING_ARTIFACTS_PATH", str(settings.models_dir / "docling"))
