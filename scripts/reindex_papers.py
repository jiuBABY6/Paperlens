"""为规范论文补建当前版本 Qdrant 向量索引。"""

import argparse
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from app.config import settings
from app.multimodal import FigureUnderstandingService
from app.repository import PaperRepository
from app.services.retrieval import HybridRetriever, INDEX_VERSION


def main() -> None:
    """默认只遍历去重注册表中的规范论文；无需重新上传 PDF。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-id",
        action="append",
        help="只重建指定论文，可重复传入；默认重建全部规范论文",
    )
    args = parser.parse_args()
    repository = PaperRepository(settings.database_path)
    retriever = HybridRetriever(settings)
    figures = FigureUnderstandingService(settings, repository)
    try:
        import sqlite3
        with sqlite3.connect(settings.database_path) as database:
            if args.paper_id:
                paper_ids = args.paper_id
            else:
                paper_ids = [
                    row[0] for row in database.execute(
                        """
                        SELECT d.paper_id
                        FROM document_registry d
                        JOIN papers p ON p.id = d.paper_id
                        WHERE p.status = 'completed'
                        ORDER BY p.created_at
                        """
                    )
                ]
        for paper_id in paper_ids:
            result = repository.get(paper_id)
            if not result:
                print(f"{paper_id}: 未找到")
                continue
            paper, card = result
            figures.enrich_paper(paper)
            indexed = retriever.index(paper)
            error = "" if indexed else (retriever.last_error or "向量索引未启用")
            paper.vector_indexed = indexed
            paper.index_version = INDEX_VERSION if indexed else 0
            paper.index_error = error
            repository.save(paper, card)
            repository.update_index_status(
                paper.id, indexed, INDEX_VERSION if indexed else 0, error
            )
            status = f"完成（index v{INDEX_VERSION}）" if indexed else f"失败：{error}"
            print(f"{paper.filename}: {status}")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
