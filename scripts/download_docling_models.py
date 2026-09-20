"""Download Docling model artifacts into demo/models/docling."""

from pathlib import Path

from docling.utils.model_downloader import download_models


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "models" / "docling"


def main() -> None:
    """Download the academic-PDF model set used by PaperLens."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # force=True 会修复上次被中断的快照；已完整存在的文件不会被应用代码重复下载。
    download_models(
        output_dir=OUTPUT_DIR,
        force=True,
        progress=True,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=True,
        with_picture_classifier=True,
        with_rapidocr=True,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granite_chart_extraction=False,
    )
    print(f"Docling models are available in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
