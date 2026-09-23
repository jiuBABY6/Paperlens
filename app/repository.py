"""SQLite 持久化层：论文、证据、图表、精读卡片和问答历史。"""

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.domain import Chunk, Figure, Paper, Sentence


def load_bbox(value: str | None) -> tuple[float, float, float, float] | None:
    """兼容 SQL NULL 和旧数据中的 JSON null。"""
    if not value:
        return None
    decoded = json.loads(value)
    return tuple(decoded) if decoded else None


class PaperRepository:
    """封装 SQLite 操作，服务重启后数据仍可恢复。"""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        """创建带行字典访问能力的数据库连接。"""
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def check_ready(self) -> bool:
        """Cheap local readiness probe without mutating user data."""
        try:
            with self._connect() as db:
                return db.execute("SELECT 1").fetchone()[0] == 1
        except sqlite3.Error:
            return False

    def _initialize(self) -> None:
        """创建应用所需表结构；重复调用不会覆盖用户数据。"""
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, title TEXT NOT NULL,
                    abstract TEXT NOT NULL, pdf_path TEXT NOT NULL, parser_name TEXT NOT NULL,
                    card_json TEXT, content_sha256 TEXT, status TEXT NOT NULL DEFAULT 'completed',
                    analysis_version INTEGER NOT NULL DEFAULT 1, error_message TEXT,
                    vector_indexed INTEGER NOT NULL DEFAULT 0,
                    index_version INTEGER NOT NULL DEFAULT 0, index_error TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, page INTEGER NOT NULL,
                    section TEXT NOT NULL, text TEXT NOT NULL, bbox_json TEXT,
                    char_start INTEGER, char_end INTEGER, sentence_ids_json TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS figures (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, page INTEGER NOT NULL,
                    kind TEXT NOT NULL, caption TEXT NOT NULL, bbox_json TEXT, image_path TEXT,
                    content TEXT NOT NULL DEFAULT '', section TEXT NOT NULL DEFAULT '',
                    nearby_text TEXT NOT NULL DEFAULT '', related_sentence_ids_json TEXT,
                    description_json TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS qa_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, paper_id TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT NOT NULL, citations_json TEXT NOT NULL,
                    answerable INTEGER, claims_json TEXT, query_plan_json TEXT,
                    retrieval_json TEXT, latency_ms REAL, model_name TEXT, status TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sentences (
                    id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, chunk_id TEXT NOT NULL,
                    page INTEGER NOT NULL, section TEXT NOT NULL, text TEXT NOT NULL,
                    bbox_json TEXT, bboxes_json TEXT, FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS document_registry (
                    content_sha256 TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL UNIQUE,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS agent_traces (
                    run_id TEXT PRIMARY KEY, paper_id TEXT NOT NULL, query TEXT NOT NULL,
                    route TEXT NOT NULL, trace_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS figure_query_cache (
                    paper_id TEXT NOT NULL, figure_id TEXT NOT NULL,
                    normalized_query TEXT NOT NULL, result_json TEXT NOT NULL,
                    cache_version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(paper_id, figure_id, normalized_query),
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '新对话',
                    summary TEXT NOT NULL DEFAULT '',
                    memory_json TEXT NOT NULL DEFAULT '{}',
                    archived_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    client_message_id TEXT,
                    resolved_question TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    trace_run_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                    UNIQUE(conversation_id, message_index),
                    UNIQUE(conversation_id, turn_index, role),
                    UNIQUE(conversation_id, client_message_id)
                );
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    user_message_id TEXT NOT NULL,
                    assistant_message_id TEXT,
                    status TEXT NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    original_question TEXT NOT NULL,
                    resolved_question TEXT,
                    result_json TEXT,
                    error_json TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(paper_id) REFERENCES papers(id),
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id),
                    FOREIGN KEY(user_message_id) REFERENCES conversation_messages(id),
                    FOREIGN KEY(assistant_message_id) REFERENCES conversation_messages(id)
                );
                CREATE TABLE IF NOT EXISTS paper_learning_memories (
                    user_scope TEXT NOT NULL DEFAULT 'local',
                    paper_id TEXT NOT NULL,
                    memory_json TEXT NOT NULL DEFAULT '{}',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(user_scope, paper_id),
                    FOREIGN KEY(paper_id) REFERENCES papers(id)
                );
                CREATE TABLE IF NOT EXISTS paper_memory_items (
                    id TEXT PRIMARY KEY,
                    user_scope TEXT NOT NULL DEFAULT 'local',
                    paper_id TEXT NOT NULL,
                    source_conversation_id TEXT,
                    source_run_id TEXT,
                    source_analysis_version INTEGER NOT NULL DEFAULT 1,
                    fingerprint TEXT NOT NULL,
                    question TEXT NOT NULL,
                    resolved_question TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    evidence_types_json TEXT NOT NULL DEFAULT '[]',
                    sections_json TEXT NOT NULL DEFAULT '[]',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    status TEXT NOT NULL DEFAULT 'active',
                    pinned INTEGER NOT NULL DEFAULT 0,
                    user_note TEXT NOT NULL DEFAULT '',
                    memory_version INTEGER NOT NULL DEFAULT 2,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed_at TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(paper_id) REFERENCES papers(id),
                    UNIQUE(user_scope, paper_id, fingerprint),
                    UNIQUE(user_scope, paper_id, source_run_id)
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_paper_updated
                    ON conversations(paper_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_turn
                    ON conversation_messages(conversation_id, turn_index, message_index);
                CREATE INDEX IF NOT EXISTS idx_runs_conversation_created
                    ON agent_runs(conversation_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_status ON agent_runs(status);
                CREATE INDEX IF NOT EXISTS idx_learning_memory_paper_updated
                    ON paper_learning_memories(paper_id, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_items_paper_status
                    ON paper_memory_items(user_scope, paper_id, status, updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_memory_items_expiry
                    ON paper_memory_items(status, pinned, expires_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_run_per_conversation
                    ON agent_runs(conversation_id) WHERE status IN ('queued', 'running');
            """)
            paper_columns = {row[1] for row in db.execute("PRAGMA table_info(papers)")}
            additions = {
                "content_sha256": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'completed'",
                "analysis_version": "INTEGER NOT NULL DEFAULT 1",
                "error_message": "TEXT",
                "updated_at": "TEXT",
                "vector_indexed": "INTEGER NOT NULL DEFAULT 0",
                "index_version": "INTEGER NOT NULL DEFAULT 0",
                "index_error": "TEXT",
            }
            for name, definition in additions.items():
                if name not in paper_columns:
                    db.execute(f"ALTER TABLE papers ADD COLUMN {name} {definition}")
            history_columns = {row[1] for row in db.execute("PRAGMA table_info(qa_history)")}
            history_additions = {
                "answerable": "INTEGER",
                "claims_json": "TEXT",
                "query_plan_json": "TEXT",
                "retrieval_json": "TEXT",
                "latency_ms": "REAL",
                "model_name": "TEXT",
                "status": "TEXT",
            }
            for name, definition in history_additions.items():
                if name not in history_columns:
                    db.execute(f"ALTER TABLE qa_history ADD COLUMN {name} {definition}")
            figure_cache_columns = {
                row[1] for row in db.execute("PRAGMA table_info(figure_query_cache)")
            }
            if "cache_version" not in figure_cache_columns:
                db.execute(
                    "ALTER TABLE figure_query_cache "
                    "ADD COLUMN cache_version INTEGER NOT NULL DEFAULT 1"
                )
            columns = {row[1] for row in db.execute("PRAGMA table_info(figures)")}
            if "content" not in columns:
                db.execute("ALTER TABLE figures ADD COLUMN content TEXT NOT NULL DEFAULT ''")
            figure_additions = {
                "section": "TEXT NOT NULL DEFAULT ''",
                "nearby_text": "TEXT NOT NULL DEFAULT ''",
                "related_sentence_ids_json": "TEXT",
                "description_json": "TEXT",
            }
            for name, definition in figure_additions.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE figures ADD COLUMN {name} {definition}")
            chunk_columns = {row[1] for row in db.execute("PRAGMA table_info(chunks)")}
            if "char_start" not in chunk_columns:
                db.execute("ALTER TABLE chunks ADD COLUMN char_start INTEGER")
            if "char_end" not in chunk_columns:
                db.execute("ALTER TABLE chunks ADD COLUMN char_end INTEGER")
            if "sentence_ids_json" not in chunk_columns:
                db.execute("ALTER TABLE chunks ADD COLUMN sentence_ids_json TEXT")
            sentence_columns = {row[1] for row in db.execute("PRAGMA table_info(sentences)")}
            if "bboxes_json" not in sentence_columns:
                db.execute("ALTER TABLE sentences ADD COLUMN bboxes_json TEXT")
            self._backfill_document_registry(db)
            db.execute(
                """
                UPDATE agent_runs
                SET status = 'interrupted', finished_at = CURRENT_TIMESTAMP,
                    error_json = ?
                WHERE status IN ('queued', 'running')
                """,
                (json.dumps({"code": "SERVER_RESTARTED", "message": "服务重启，运行已中断。"}, ensure_ascii=False),),
            )

    def _backfill_document_registry(self, db: sqlite3.Connection) -> None:
        """为旧数据计算哈希，并将证据最完整的新版本登记为规范记录。"""
        rows = db.execute(
            """
            SELECT p.id, p.pdf_path, p.content_sha256, p.status, p.created_at,
                   (SELECT COUNT(*) FROM sentences s WHERE s.paper_id = p.id) AS sentence_count
            FROM papers p ORDER BY p.created_at, p.id
            """
        ).fetchall()
        groups: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            digest = row["content_sha256"]
            path = Path(row["pdf_path"])
            if not digest and path.is_file():
                digest = self._sha256(path)
                db.execute("UPDATE papers SET content_sha256 = ? WHERE id = ?", (digest, row["id"]))
            if digest:
                groups.setdefault(digest, []).append(row)
        for digest, candidates in groups.items():
            canonical = max(candidates, key=lambda row: (
                row["status"] == "completed",
                int(row["sentence_count"] or 0),
                str(row["created_at"] or ""),
                row["id"],
            ))
            db.execute(
                """
                INSERT INTO document_registry(content_sha256, paper_id) VALUES (?, ?)
                ON CONFLICT(content_sha256) DO UPDATE SET paper_id = excluded.paper_id
                """,
                (digest, canonical["id"]),
            )

    def _sha256(self, path: Path) -> str:
        """分块计算文件摘要，避免一次性加载整篇论文。"""
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    def reserve_upload(
        self,
        paper_id: str,
        filename: str,
        pdf_path: Path,
        content_sha256: str,
    ) -> str | None:
        """原子登记文件哈希；若已存在则返回规范论文 ID。"""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT paper_id FROM document_registry WHERE content_sha256 = ?",
                (content_sha256,),
            ).fetchone()
            if existing:
                return str(existing["paper_id"])
            db.execute(
                """
                INSERT INTO papers(
                    id, filename, title, abstract, pdf_path, parser_name, card_json,
                    content_sha256, status, analysis_version, error_message, updated_at
                ) VALUES (?, ?, '', '', ?, 'pending', NULL, ?, 'processing', 1, NULL, CURRENT_TIMESTAMP)
                """,
                (paper_id, filename, str(pdf_path), content_sha256),
            )
            db.execute(
                "INSERT INTO document_registry(content_sha256, paper_id) VALUES (?, ?)",
                (content_sha256, paper_id),
            )
        return None

    def release_upload(self, paper_id: str, content_sha256: str) -> None:
        """首次处理失败时释放哈希和占位记录，使同一文件可以重试。"""
        with self._connect() as db:
            db.execute(
                "DELETE FROM document_registry WHERE content_sha256 = ? AND paper_id = ?",
                (content_sha256, paper_id),
            )
            db.execute("DELETE FROM papers WHERE id = ? AND status = 'processing'", (paper_id,))

    def get_status(self, paper_id: str) -> str | None:
        """返回上传或重新分析状态。"""
        with self._connect() as db:
            row = db.execute("SELECT status FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return str(row["status"]) if row else None

    def begin_reanalysis(self, paper_id: str) -> int:
        """将已完成论文切换为重新分析状态并递增版本号。"""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, analysis_version FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            if not row:
                raise KeyError(paper_id)
            if row["status"] == "processing":
                raise RuntimeError("论文正在处理中")
            version = int(row["analysis_version"] or 1) + 1
            db.execute(
                "UPDATE papers SET status = 'processing', error_message = NULL, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (paper_id,),
            )
        return version

    def finish_reanalysis_failure(self, paper_id: str, message: str) -> None:
        """重新分析失败时保留上一版本数据并恢复可读取状态。"""
        with self._connect() as db:
            db.execute(
                "UPDATE papers SET status = 'completed', error_message = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (message[:1000], paper_id),
            )

    def save(self, paper: Paper, card: dict | None) -> None:
        """以事务方式保存论文及全部关联内容。"""
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO papers(
                    id, filename, title, abstract, pdf_path, parser_name, card_json,
                    content_sha256, status, analysis_version, error_message,
                    vector_indexed, index_version, index_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'completed', ?, NULL, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    filename = excluded.filename,
                    title = excluded.title,
                    abstract = excluded.abstract,
                    pdf_path = excluded.pdf_path,
                    parser_name = excluded.parser_name,
                    card_json = excluded.card_json,
                    content_sha256 = excluded.content_sha256,
                    status = 'completed',
                    analysis_version = excluded.analysis_version,
                    error_message = NULL,
                    vector_indexed = excluded.vector_indexed,
                    index_version = excluded.index_version,
                    index_error = excluded.index_error,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    paper.id, paper.filename, paper.title, paper.abstract, paper.pdf_path,
                    paper.parser_name, json.dumps(card, ensure_ascii=False) if card else None,
                    paper.content_sha256 or None, paper.analysis_version,
                    int(paper.vector_indexed), paper.index_version, paper.index_error or None,
                ),
            )
            # save 只在首次建库、重新分析或显式重建索引时调用。此时 Figure
            # 的图片、章节或离线描述可能已经变化，旧 query-conditioned 结果不可复用。
            db.execute("DELETE FROM figure_query_cache WHERE paper_id = ?", (paper.id,))
            db.execute("DELETE FROM sentences WHERE paper_id = ?", (paper.id,))
            db.execute("DELETE FROM chunks WHERE paper_id = ?", (paper.id,))
            db.execute("DELETE FROM figures WHERE paper_id = ?", (paper.id,))
            db.executemany("INSERT INTO chunks(id, paper_id, page, section, text, bbox_json, char_start, char_end, sentence_ids_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (item.id, item.paper_id, item.page, item.section, item.text, json.dumps(item.bbox) if item.bbox else None, item.char_start, item.char_end, json.dumps(item.sentence_ids))
                for item in paper.chunks
            ])
            db.executemany("INSERT INTO sentences(id, paper_id, chunk_id, page, section, text, bbox_json, bboxes_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
                (item.id, item.paper_id, item.chunk_id, item.page, item.section, item.text, json.dumps(item.bbox) if item.bbox else None, json.dumps(item.bboxes))
                for item in paper.sentences
            ])
            db.executemany("INSERT INTO figures(id, paper_id, page, kind, caption, bbox_json, image_path, content, section, nearby_text, related_sentence_ids_json, description_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                (item.id, item.paper_id, item.page, item.kind, item.caption, json.dumps(item.bbox) if item.bbox else None, item.image_path, item.content, item.section, item.nearby_text, json.dumps(item.related_sentence_ids), json.dumps(item.description, ensure_ascii=False))
                for item in paper.figures
            ])

    def get(self, paper_id: str) -> tuple[Paper, dict | None] | None:
        """读取论文、证据、图表及其已生成精读卡片。"""
        with self._connect() as db:
            row = db.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
            if not row:
                return None
            chunks = [Chunk(item["id"], item["paper_id"], item["page"], item["section"], item["text"], load_bbox(item["bbox_json"]), item["char_start"] if "char_start" in item.keys() else None, item["char_end"] if "char_end" in item.keys() else None, json.loads(item["sentence_ids_json"]) if item["sentence_ids_json"] else []) for item in db.execute("SELECT * FROM chunks WHERE paper_id = ?", (paper_id,))]
            sentences = [Sentence(item["id"], item["paper_id"], item["chunk_id"], item["page"], item["section"], item["text"], load_bbox(item["bbox_json"]), [tuple(box) for box in json.loads(item["bboxes_json"])] if item["bboxes_json"] else []) for item in db.execute("SELECT * FROM sentences WHERE paper_id = ?", (paper_id,))]
            figures = [Figure(
                item["id"], item["paper_id"], item["page"], item["kind"], item["caption"],
                load_bbox(item["bbox_json"]), item["image_path"],
                item["content"] if "content" in item.keys() else "",
                item["section"] if "section" in item.keys() else "",
                item["nearby_text"] if "nearby_text" in item.keys() else "",
                json.loads(item["related_sentence_ids_json"] or "[]") if "related_sentence_ids_json" in item.keys() else [],
                json.loads(item["description_json"] or "{}") if "description_json" in item.keys() else {},
            ) for item in db.execute("SELECT * FROM figures WHERE paper_id = ?", (paper_id,))]
        if not sentences:
            sentences = [Sentence(chunk.id, chunk.paper_id, chunk.id, chunk.page, chunk.section, chunk.text, chunk.bbox) for chunk in chunks]
        paper = Paper(
            row["id"], row["filename"], row["title"], row["abstract"],
            row["pdf_path"], row["parser_name"], chunks, figures, sentences,
            row["content_sha256"] if "content_sha256" in row.keys() else "",
            row["status"] if "status" in row.keys() else "completed",
            int(row["analysis_version"] or 1) if "analysis_version" in row.keys() else 1,
            bool(row["vector_indexed"]) if "vector_indexed" in row.keys() else False,
            int(row["index_version"] or 0) if "index_version" in row.keys() else 0,
            str(row["index_error"] or "") if "index_error" in row.keys() else "",
        )
        return paper, json.loads(row["card_json"]) if row["card_json"] else None

    def save_answer(
        self,
        paper_id: str,
        question: str,
        answer: str,
        citations: list[dict],
        *,
        answerable: bool | None = None,
        claims: list[dict] | None = None,
        query_plan: dict | None = None,
        retrieval: dict | None = None,
        latency_ms: float | None = None,
        model_name: str = "",
        status: str = "ok",
    ) -> None:
        """保存可复现的端到端问答轨迹。"""
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO qa_history(
                    paper_id, question, answer, citations_json, answerable, claims_json,
                    query_plan_json, retrieval_json, latency_ms, model_name, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    question,
                    answer,
                    json.dumps(citations, ensure_ascii=False),
                    None if answerable is None else int(answerable),
                    json.dumps(claims or [], ensure_ascii=False),
                    json.dumps(query_plan or {}, ensure_ascii=False),
                    json.dumps(retrieval or {}, ensure_ascii=False),
                    latency_ms,
                    model_name,
                    status,
                ),
            )

    def update_index_status(
        self,
        paper_id: str,
        indexed: bool,
        index_version: int,
        error: str = "",
    ) -> None:
        """在不重写解析结果的情况下持久化向量索引状态。"""
        with self._connect() as db:
            db.execute(
                """
                UPDATE papers
                SET vector_indexed = ?, index_version = ?, index_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (int(indexed), index_version, error[:1000] or None, paper_id),
            )

    def save_trace(self, paper_id: str, trace: dict) -> None:
        """持久化一次完整 Standard/Agentic RAG 轨迹。"""
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO agent_traces(run_id, paper_id, query, route, trace_json) VALUES (?, ?, ?, ?, ?)",
                (trace["run_id"], paper_id, trace.get("query", ""), trace.get("route", ""), json.dumps(trace, ensure_ascii=False)),
            )

    def get_trace(self, paper_id: str, run_id: str) -> dict | None:
        """读取属于指定论文的 Agent Trace。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT trace_json FROM agent_traces WHERE run_id = ? AND paper_id = ?",
                (run_id, paper_id),
            ).fetchone()
        return json.loads(row["trace_json"]) if row else None

    def get_figure_query_cache(
        self,
        paper_id: str,
        figure_id: str,
        normalized_query: str,
        cache_version: int = 1,
    ) -> dict | None:
        """读取 query-conditioned figure understanding 缓存。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT result_json FROM figure_query_cache "
                "WHERE paper_id = ? AND figure_id = ? AND normalized_query = ? "
                "AND cache_version = ?",
                (paper_id, figure_id, normalized_query, cache_version),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def save_figure_query_cache(
        self,
        paper_id: str,
        figure_id: str,
        normalized_query: str,
        result: dict,
        cache_version: int = 1,
    ) -> None:
        """保存按 figure_id + normalized_query 唯一的视觉阅读结果。"""
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO figure_query_cache("
                "paper_id, figure_id, normalized_query, result_json, cache_version"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    paper_id,
                    figure_id,
                    normalized_query,
                    json.dumps(result, ensure_ascii=False),
                    cache_version,
                ),
            )

    def delete_figure_query_cache(
        self,
        paper_id: str,
        figure_id: str | None = None,
    ) -> int:
        """删除指定论文或单张 Figure 的查询缓存，并返回删除行数。"""
        with self._connect() as db:
            if figure_id:
                cursor = db.execute(
                    "DELETE FROM figure_query_cache WHERE paper_id = ? AND figure_id = ?",
                    (paper_id, figure_id),
                )
            else:
                cursor = db.execute(
                    "DELETE FROM figure_query_cache WHERE paper_id = ?",
                    (paper_id,),
                )
        return max(cursor.rowcount, 0)

    @staticmethod
    def _json(value: str | None, fallback: Any) -> Any:
        """Safely decode JSON fields written by the conversation subsystem."""
        if not value:
            return fallback
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def create_conversation(
        self, conversation_id: str, paper_id: str, title: str = "新对话"
    ) -> dict[str, Any]:
        """Create a conversation bound permanently to one completed paper."""
        with self._connect() as db:
            paper = db.execute(
                "SELECT status FROM papers WHERE id = ?", (paper_id,)
            ).fetchone()
            if not paper:
                raise KeyError("paper_not_found")
            if paper["status"] != "completed":
                raise RuntimeError("paper_not_ready")
            db.execute(
                "INSERT INTO conversations(id, paper_id, title) VALUES (?, ?, ?)",
                (conversation_id, paper_id, title.strip()[:120] or "新对话"),
            )
        return self.get_conversation(paper_id, conversation_id, include_messages=False)  # type: ignore[return-value]

    def list_conversations(
        self, paper_id: str, *, include_archived: bool = False
    ) -> list[dict[str, Any]]:
        """List one paper's conversations without leaking conversations of another paper."""
        where = "paper_id = ?" if include_archived else "paper_id = ? AND archived_at IS NULL"
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT c.*,
                       (SELECT content FROM conversation_messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.message_index DESC LIMIT 1) AS last_message,
                       (SELECT COUNT(*) FROM conversation_messages m
                        WHERE m.conversation_id = c.id AND m.role = 'user') AS turn_count
                FROM conversations c WHERE {where}
                ORDER BY c.updated_at DESC, c.created_at DESC
                """,
                (paper_id,),
            ).fetchall()
        return [self._conversation_payload(row) for row in rows]

    def get_conversation(
        self, paper_id: str, conversation_id: str, *, include_messages: bool = True
    ) -> dict[str, Any] | None:
        """Load a scoped conversation and optionally its full ordered timeline."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM conversations WHERE id = ? AND paper_id = ?",
                (conversation_id, paper_id),
            ).fetchone()
            if not row:
                return None
            payload = self._conversation_payload(row)
            if include_messages:
                messages = db.execute(
                    "SELECT * FROM conversation_messages WHERE conversation_id = ? ORDER BY message_index",
                    (conversation_id,),
                ).fetchall()
                payload["messages"] = [self._message_payload(item) for item in messages]
        return payload

    def update_conversation(
        self, paper_id: str, conversation_id: str, *, title: str | None = None,
        summary: str | None = None, memory: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """Update user-visible title or derived memory while preserving paper ownership."""
        assignments: list[str] = []
        values: list[Any] = []
        if title is not None:
            assignments.append("title = ?")
            values.append(title.strip()[:120] or "新对话")
        if summary is not None:
            assignments.append("summary = ?")
            values.append(summary[:12000])
        if memory is not None:
            assignments.append("memory_json = ?")
            values.append(json.dumps(memory, ensure_ascii=False))
        if assignments:
            assignments.append("updated_at = CURRENT_TIMESTAMP")
            with self._connect() as db:
                cursor = db.execute(
                    f"UPDATE conversations SET {', '.join(assignments)} WHERE id = ? AND paper_id = ?",
                    (*values, conversation_id, paper_id),
                )
                if cursor.rowcount == 0:
                    return None
        return self.get_conversation(paper_id, conversation_id, include_messages=False)

    def archive_conversation(self, paper_id: str, conversation_id: str) -> bool:
        """Soft-delete a conversation; active runs must be cancelled first."""
        with self._connect() as db:
            active = db.execute(
                "SELECT 1 FROM agent_runs WHERE conversation_id = ? AND status IN ('queued', 'running')",
                (conversation_id,),
            ).fetchone()
            if active:
                raise RuntimeError("active_run")
            cursor = db.execute(
                "UPDATE conversations SET archived_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND paper_id = ? AND archived_at IS NULL",
                (conversation_id, paper_id),
            )
        return cursor.rowcount > 0

    def get_paper_learning_memory(
        self, paper_id: str, *, user_scope: str = "local"
    ) -> dict[str, Any]:
        """Read paper-scoped long-term learning memory without mixing papers/users."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM paper_learning_memories WHERE user_scope = ? AND paper_id = ?",
                (user_scope, paper_id),
            ).fetchone()
        if not row:
            return {
                "user_scope": user_scope,
                "paper_id": paper_id,
                "version": 1,
                "memory": {},
                "created_at": None,
                "updated_at": None,
            }
        return {
            "user_scope": row["user_scope"],
            "paper_id": row["paper_id"],
            "version": int(row["version"]),
            "memory": self._json(row["memory_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def update_paper_learning_memory(
        self,
        paper_id: str,
        memory: dict[str, Any],
        *,
        user_scope: str = "local",
        version: int = 1,
    ) -> dict[str, Any]:
        """Upsert bounded derived learning state; source messages remain immutable."""
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,)).fetchone():
                raise KeyError("paper_not_found")
            db.execute(
                """
                INSERT INTO paper_learning_memories(
                    user_scope, paper_id, memory_json, version
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(user_scope, paper_id) DO UPDATE SET
                    memory_json = excluded.memory_json,
                    version = excluded.version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_scope,
                    paper_id,
                    json.dumps(memory, ensure_ascii=False),
                    max(1, int(version)),
                ),
            )
        return self.get_paper_learning_memory(paper_id, user_scope=user_scope)

    def clear_paper_learning_memory(
        self, paper_id: str, *, user_scope: str = "local"
    ) -> bool:
        """Delete only derived learning memory, never source papers or conversations."""
        with self._connect() as db:
            aggregate = db.execute(
                "DELETE FROM paper_learning_memories WHERE user_scope = ? AND paper_id = ?",
                (user_scope, paper_id),
            )
            items = db.execute(
                "DELETE FROM paper_memory_items WHERE user_scope = ? AND paper_id = ?",
                (user_scope, paper_id),
            )
        return aggregate.rowcount > 0 or items.rowcount > 0

    def upsert_memory_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Persist one versioned memory item, idempotent by source run and fingerprint."""
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO paper_memory_items(
                    id, user_scope, paper_id, source_conversation_id, source_run_id,
                    source_analysis_version, fingerprint, question, resolved_question,
                    evidence_ids_json, evidence_types_json, sections_json,
                    importance, confidence, status, pinned, user_note,
                    memory_version, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_scope, paper_id, fingerprint) DO UPDATE SET
                    source_conversation_id = excluded.source_conversation_id,
                    source_run_id = excluded.source_run_id,
                    source_analysis_version = excluded.source_analysis_version,
                    question = excluded.question,
                    resolved_question = excluded.resolved_question,
                    evidence_ids_json = excluded.evidence_ids_json,
                    evidence_types_json = excluded.evidence_types_json,
                    sections_json = excluded.sections_json,
                    importance = MAX(paper_memory_items.importance, excluded.importance),
                    confidence = MAX(paper_memory_items.confidence, excluded.confidence),
                    status = CASE
                        WHEN paper_memory_items.status = 'forgotten' THEN 'forgotten'
                        ELSE excluded.status END,
                    memory_version = excluded.memory_version,
                    expires_at = CASE
                        WHEN paper_memory_items.pinned = 1 THEN NULL
                        ELSE excluded.expires_at END,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    item["id"], item.get("user_scope", "local"), item["paper_id"],
                    item.get("source_conversation_id"), item.get("source_run_id"),
                    int(item.get("source_analysis_version", 1)), item["fingerprint"],
                    item.get("question", ""), item.get("resolved_question", ""),
                    json.dumps(item.get("evidence_ids", []), ensure_ascii=False),
                    json.dumps(item.get("evidence_types", []), ensure_ascii=False),
                    json.dumps(item.get("sections", []), ensure_ascii=False),
                    float(item.get("importance", 0.5)),
                    float(item.get("confidence", 1.0)), item.get("status", "active"),
                    int(bool(item.get("pinned", False))), item.get("user_note", "")[:2000],
                    int(item.get("memory_version", 2)), item.get("expires_at"),
                ),
            )
            row = db.execute(
                "SELECT * FROM paper_memory_items WHERE user_scope = ? AND paper_id = ? AND fingerprint = ?",
                (item.get("user_scope", "local"), item["paper_id"], item["fingerprint"]),
            ).fetchone()
        return self._memory_item_payload(row)

    def list_memory_items(
        self, paper_id: str, *, user_scope: str = "local",
        include_inactive: bool = False, limit: int = 200,
    ) -> list[dict[str, Any]]:
        where = "user_scope = ? AND paper_id = ?"
        if not include_inactive:
            where += " AND status IN ('active', 'unresolved')"
        with self._connect() as db:
            rows = db.execute(
                f"""
                SELECT * FROM paper_memory_items WHERE {where}
                ORDER BY pinned DESC, importance DESC, updated_at DESC LIMIT ?
                """,
                (user_scope, paper_id, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [self._memory_item_payload(row) for row in rows]

    def get_memory_item(
        self, paper_id: str, memory_id: str, *, user_scope: str = "local"
    ) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM paper_memory_items WHERE id = ? AND user_scope = ? AND paper_id = ?",
                (memory_id, user_scope, paper_id),
            ).fetchone()
        return self._memory_item_payload(row) if row else None

    def update_memory_item(
        self, paper_id: str, memory_id: str, *, user_scope: str = "local",
        pinned: bool | None = None, user_note: str | None = None,
        status: str | None = None, importance: float | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any] | None:
        allowed_statuses = {"active", "unresolved", "stale", "archived", "forgotten"}
        assignments: list[str] = []
        values: list[Any] = []
        if pinned is not None:
            flag = int(bool(pinned))
            assignments.append("pinned = ?")
            values.append(flag)
            if flag:
                assignments.append("expires_at = NULL")
        if user_note is not None:
            assignments.append("user_note = ?")
            values.append(user_note.strip()[:2000])
        if status is not None:
            if status not in allowed_statuses:
                raise ValueError("invalid_memory_status")
            assignments.append("status = ?")
            values.append(status)
        if importance is not None:
            assignments.append("importance = ?")
            values.append(max(0.0, min(float(importance), 1.0)))
        if expires_at is not None:
            assignments.append("expires_at = ?")
            values.append(expires_at)
        if not assignments:
            return self.get_memory_item(paper_id, memory_id, user_scope=user_scope)
        assignments.append("updated_at = CURRENT_TIMESTAMP")
        with self._connect() as db:
            cursor = db.execute(
                f"UPDATE paper_memory_items SET {', '.join(assignments)} "
                "WHERE id = ? AND user_scope = ? AND paper_id = ?",
                (*values, memory_id, user_scope, paper_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_memory_item(paper_id, memory_id, user_scope=user_scope)

    def mark_memory_stale_for_version(
        self, paper_id: str, analysis_version: int, *, user_scope: str = "local"
    ) -> int:
        """Invalidate derived memory whose Evidence belongs to an older parse version."""
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE paper_memory_items SET status = 'stale', updated_at = CURRENT_TIMESTAMP
                WHERE user_scope = ? AND paper_id = ? AND status = 'active'
                  AND source_analysis_version <> ?
                """,
                (user_scope, paper_id, int(analysis_version)),
            )
        return max(cursor.rowcount, 0)

    def memory_status_counts(self, *, user_scope: str = "local") -> dict[str, int]:
        """Return low-cardinality global memory counts for operational metrics."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT status, COUNT(*) AS item_count
                FROM paper_memory_items WHERE user_scope = ? GROUP BY status
                """,
                (user_scope,),
            ).fetchall()
        return {str(row["status"]): int(row["item_count"]) for row in rows}

    def expire_memory_items(self, *, user_scope: str = "local") -> int:
        with self._connect() as db:
            cursor = db.execute(
                """
                UPDATE paper_memory_items SET status = 'archived', updated_at = CURRENT_TIMESTAMP
                WHERE user_scope = ? AND status IN ('active', 'unresolved')
                  AND pinned = 0 AND expires_at IS NOT NULL
                  AND expires_at <= CURRENT_TIMESTAMP
                """,
                (user_scope,),
            )
        return max(cursor.rowcount, 0)

    def list_memory_backfill_candidates(
        self, paper_id: str, *, user_scope: str = "local"
    ) -> list[dict[str, Any]]:
        """Return completed historical turns not yet represented by source_run_id."""
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT r.id AS run_id, r.conversation_id, r.original_question,
                       r.resolved_question, r.status AS run_status, r.result_json,
                       r.finished_at, m.citations_json, m.metadata_json,
                       p.analysis_version
                FROM agent_runs r
                JOIN papers p ON p.id = r.paper_id
                LEFT JOIN conversation_messages m ON m.id = r.assistant_message_id
                LEFT JOIN paper_memory_items memory
                  ON memory.user_scope = ? AND memory.paper_id = r.paper_id
                 AND memory.source_run_id = r.id
                WHERE r.paper_id = ? AND r.status IN ('completed', 'partial')
                  AND memory.id IS NULL
                ORDER BY r.created_at
                """,
                (user_scope, paper_id),
            ).fetchall()
        return [{
            "run_id": row["run_id"],
            "conversation_id": row["conversation_id"],
            "original_question": row["original_question"],
            "resolved_question": row["resolved_question"] or row["original_question"],
            "run_status": row["run_status"],
            "result": self._json(row["result_json"], {}),
            "citations": self._json(row["citations_json"], []),
            "metadata": self._json(row["metadata_json"], {}),
            "analysis_version": int(row["analysis_version"] or 1),
            "finished_at": row["finished_at"],
        } for row in rows]

    def create_message_and_run(
        self, *, paper_id: str, conversation_id: str, question: str,
        client_message_id: str, message_id: str, run_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Atomically create a user turn and queued run, with browser retry idempotency."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            conversation = db.execute(
                "SELECT archived_at FROM conversations WHERE id = ? AND paper_id = ?",
                (conversation_id, paper_id),
            ).fetchone()
            if not conversation:
                raise KeyError("conversation_not_found")
            if conversation["archived_at"]:
                raise RuntimeError("conversation_archived")
            existing = db.execute(
                """
                SELECT m.*, r.id AS run_id, r.status AS run_status
                FROM conversation_messages m
                JOIN agent_runs r ON r.user_message_id = m.id
                WHERE m.conversation_id = ? AND m.client_message_id = ?
                """,
                (conversation_id, client_message_id),
            ).fetchone()
            if existing:
                return self._message_payload(existing), {
                    "id": existing["run_id"], "status": existing["run_status"],
                    "paper_id": paper_id, "conversation_id": conversation_id,
                }, False
            if db.execute(
                "SELECT 1 FROM agent_runs WHERE conversation_id = ? AND status IN ('queued', 'running')",
                (conversation_id,),
            ).fetchone():
                raise RuntimeError("active_run")
            counters = db.execute(
                """
                SELECT COALESCE(MAX(turn_index), 0) AS turn_index,
                       COALESCE(MAX(message_index), 0) AS message_index
                FROM conversation_messages WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            turn_index = int(counters["turn_index"]) + 1
            message_index = int(counters["message_index"]) + 1
            db.execute(
                """
                INSERT INTO conversation_messages(
                    id, conversation_id, turn_index, message_index, role, content,
                    status, client_message_id
                ) VALUES (?, ?, ?, ?, 'user', ?, 'completed', ?)
                """,
                (message_id, conversation_id, turn_index, message_index, question, client_message_id),
            )
            db.execute(
                """
                INSERT INTO agent_runs(
                    id, paper_id, conversation_id, user_message_id, status, original_question
                ) VALUES (?, ?, ?, ?, 'queued', ?)
                """,
                (run_id, paper_id, conversation_id, message_id, question),
            )
            db.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )
            message = db.execute(
                "SELECT * FROM conversation_messages WHERE id = ?", (message_id,)
            ).fetchone()
            run = db.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
        return self._message_payload(message), self._run_payload(run), True

    def get_run(self, paper_id: str, conversation_id: str, run_id: str) -> dict[str, Any] | None:
        """Load a run only when all three ownership identifiers match."""
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM agent_runs WHERE id = ? AND paper_id = ? AND conversation_id = ?",
                (run_id, paper_id, conversation_id),
            ).fetchone()
        return self._run_payload(row) if row else None

    def mark_run_running(self, paper_id: str, conversation_id: str, run_id: str) -> bool:
        """Transition queued -> running; terminal runs are immutable."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE agent_runs SET status = 'running', started_at = CURRENT_TIMESTAMP "
                "WHERE id = ? AND paper_id = ? AND conversation_id = ? AND status = 'queued'",
                (run_id, paper_id, conversation_id),
            )
        return cursor.rowcount > 0

    def request_run_cancel(self, paper_id: str, conversation_id: str, run_id: str) -> bool:
        """Request cooperative cancellation without mutating an already terminal run."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE agent_runs SET cancel_requested = 1 WHERE id = ? AND paper_id = ? "
                "AND conversation_id = ? AND status IN ('queued', 'running')",
                (run_id, paper_id, conversation_id),
            )
        return cursor.rowcount > 0

    def is_cancel_requested(self, paper_id: str, conversation_id: str, run_id: str) -> bool:
        run = self.get_run(paper_id, conversation_id, run_id)
        return bool(run and run["cancel_requested"])

    def finish_conversation_run(
        self, *, paper_id: str, conversation_id: str, run_id: str,
        status: str, answer: str = "", resolved_question: str = "",
        citations: list[dict] | None = None, metadata: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None, error: dict[str, Any] | None = None,
        assistant_message_id: str | None = None
    ) -> dict[str, Any]:
        """Atomically persist the final assistant message and terminal run state."""
        terminal = {"completed", "partial", "failed", "cancelled", "timed_out", "interrupted"}
        if status not in terminal:
            raise ValueError(f"invalid terminal run status: {status}")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            run = db.execute(
                "SELECT * FROM agent_runs WHERE id = ? AND paper_id = ? AND conversation_id = ?",
                (run_id, paper_id, conversation_id),
            ).fetchone()
            if not run:
                raise KeyError("run_not_found")
            if run["status"] not in {"queued", "running"}:
                return self._run_payload(run)
            user = db.execute(
                "SELECT * FROM conversation_messages WHERE id = ?", (run["user_message_id"],)
            ).fetchone()
            if answer or status in {"completed", "partial"}:
                assistant_message_id = assistant_message_id or f"assistant-{run_id}"
                db.execute(
                    """
                    INSERT INTO conversation_messages(
                        id, conversation_id, turn_index, message_index, role, content,
                        status, resolved_question, citations_json, metadata_json, trace_run_id
                    ) VALUES (?, ?, ?, ?, 'assistant', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assistant_message_id, conversation_id, user["turn_index"],
                        int(user["message_index"]) + 1, answer, status, resolved_question,
                        json.dumps(citations or [], ensure_ascii=False),
                        json.dumps(metadata or {}, ensure_ascii=False), run_id,
                    ),
                )
            db.execute(
                """
                UPDATE agent_runs SET status = ?, assistant_message_id = ?,
                    resolved_question = ?, result_json = ?, error_json = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    status, assistant_message_id, resolved_question or None,
                    json.dumps(result or {}, ensure_ascii=False),
                    json.dumps(error or {}, ensure_ascii=False) if error else None, run_id,
                ),
            )
            db.execute(
                "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (conversation_id,),
            )
            final = db.execute("SELECT * FROM agent_runs WHERE id = ?", (run_id,)).fetchone()
        return self._run_payload(final)

    @classmethod
    def _conversation_payload(cls, row: sqlite3.Row) -> dict[str, Any]:
        keys = set(row.keys())
        return {
            "id": row["id"], "paper_id": row["paper_id"], "title": row["title"],
            "summary": row["summary"], "memory": cls._json(row["memory_json"], {}),
            "archived_at": row["archived_at"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_message": row["last_message"] if "last_message" in keys else None,
            "turn_count": int(row["turn_count"] or 0) if "turn_count" in keys else None,
        }

    @classmethod
    def _message_payload(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "conversation_id": row["conversation_id"],
            "turn_index": int(row["turn_index"]), "message_index": int(row["message_index"]),
            "role": row["role"], "content": row["content"], "status": row["status"],
            "client_message_id": row["client_message_id"],
            "resolved_question": row["resolved_question"],
            "citations": cls._json(row["citations_json"], []),
            "metadata": cls._json(row["metadata_json"], {}),
            "trace_run_id": row["trace_run_id"], "created_at": row["created_at"],
        }

    @classmethod
    def _run_payload(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "paper_id": row["paper_id"],
            "conversation_id": row["conversation_id"],
            "user_message_id": row["user_message_id"],
            "assistant_message_id": row["assistant_message_id"], "status": row["status"],
            "cancel_requested": bool(row["cancel_requested"]),
            "original_question": row["original_question"],
            "resolved_question": row["resolved_question"],
            "result": cls._json(row["result_json"], {}),
            "error": cls._json(row["error_json"], {}),
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "created_at": row["created_at"],
        }

    @classmethod
    def _memory_item_payload(cls, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"], "user_scope": row["user_scope"],
            "paper_id": row["paper_id"],
            "source_conversation_id": row["source_conversation_id"],
            "source_run_id": row["source_run_id"],
            "source_analysis_version": int(row["source_analysis_version"]),
            "fingerprint": row["fingerprint"], "question": row["question"],
            "resolved_question": row["resolved_question"],
            "evidence_ids": cls._json(row["evidence_ids_json"], []),
            "evidence_types": cls._json(row["evidence_types_json"], []),
            "sections": cls._json(row["sections_json"], []),
            "importance": float(row["importance"]),
            "confidence": float(row["confidence"]), "status": row["status"],
            "pinned": bool(row["pinned"]), "user_note": row["user_note"],
            "memory_version": int(row["memory_version"]),
            "access_count": int(row["access_count"]),
            "last_accessed_at": row["last_accessed_at"],
            "expires_at": row["expires_at"], "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
