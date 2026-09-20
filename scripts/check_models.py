"""检查 PaperLens 的本地模型文件，不访问网络也不加载大模型。"""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_DIR / "models"


def exists(relative_path: str) -> bool:
    """判断一个模型权重文件是否已完整落盘。"""
    return (MODELS_DIR / relative_path).is_file()


def main() -> None:
    """输出检索与 Docling 模型的可用状态，并以非零退出码提示不完整下载。"""
    docling_root = MODELS_DIR / "docling"
    # 本项目下载器请求 layout、tableformer、code/formula、图片分类和 RapidOCR 五类资产。
    # layout 可能包含 PyTorch 与 ONNX 两个目录，因此以至少五个含实际文件的目录作为完整条件。
    docling_sets = [
        path for path in docling_root.iterdir() if path.is_dir()
        and any(file.is_file() for file in path.rglob("*") if ".cache" not in file.parts)
    ] if docling_root.exists() else []
    checks = {
        "BAAI/bge-m3": exists("modelscope/bge-m3/pytorch_model.bin"),
        "BAAI/bge-reranker-v2-m3": exists("modelscope/bge-reranker-v2-m3/model.safetensors"),
        "Docling academic-PDF set": len(docling_sets) >= 5,
    }
    incomplete = list(docling_root.rglob("*.incomplete")) + list(docling_root.rglob("*.lock")) if docling_root.exists() else []
    for name, ready in checks.items():
        print(f"{'OK' if ready else 'MISSING'}  {name}")
    print(f"INFO  Docling model directories: {len(docling_sets)}")
    print(f"{'OK' if not incomplete else 'INCOMPLETE'}  Docling temporary files: {len(incomplete)}")
    if not all(checks.values()) or incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
