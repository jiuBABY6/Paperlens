"""用 ModelScope 预下载检索模型到项目目录。"""

from pathlib import Path

from modelscope import snapshot_download


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "modelscope"


def download(model_id: str, target_name: str) -> None:
    """下载一个 ModelScope 模型快照到 demo/models/modelscope 指定目录。"""
    target = MODEL_DIR / target_name
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(model_id, local_dir=str(target))
    print(f"已缓存 {model_id}: {path}")


if __name__ == "__main__":
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    download("BAAI/bge-m3", "bge-m3")
    download("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3")
