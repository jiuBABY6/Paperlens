from pathlib import Path


def test_three_column_conversation_workspace_and_external_assets() -> None:
    root = Path(__file__).parents[1] / "app" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    css = (root / "styles.css").read_text(encoding="utf-8")
    javascript = (root / "app.js").read_text(encoding="utf-8")
    assert "历史对话" in html and "阅读卡片" in html and "当前聊天" in html
    assert "conversation-column" in html and "reading-column" in html and "chat-column" in html
    assert "grid-template-columns: 260px" in css
    assert "new EventSource" in javascript
    assert "escapeHtml" in javascript
    assert "textContent +=" in javascript
