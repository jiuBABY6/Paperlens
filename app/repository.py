"""SQLite 持久化层：论文、证据、图表、精读卡片和问答历史。"""

import hashlib
import json
import sqlite3
from pathlib import Path

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
        return connection

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
