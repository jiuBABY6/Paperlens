"""集中封装 Qwen-VL OpenAI-compatible 调用。"""

import base64
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import threading

import httpx
from PIL import Image, ImageOps

from app.multi_agent.recovery import call_with_retry


class QwenVLClient:
    MIN_FIGURE_WIDTH = 1024
    MIN_FIGURE_HEIGHT = 768
    MAX_FIGURE_DIMENSION = 3072

    def __init__(self, settings) -> None:
        self.settings = settings
        self.token_usage = 0
        self.last_image_preparation: dict = {}
        self._request_lock = threading.RLock()
        self._counter_lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.settings.qwen_vl_key)

    def analyze(self, image_path: str, prompt: str) -> dict:
        """发送单张原始 Figure，并要求服务端返回 JSON object。"""
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Figure image 不存在：{path.name}")
        mime, image_bytes = self._prepare_image(path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        body = {
            "model": self.settings.qwen_vl_model,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
        }
        def request():
            with httpx.Client(timeout=120, trust_env=False) as client:
                response = client.post(
                    self.settings.qwen_vl_url,
                    headers={"Authorization": f"Bearer {self.settings.qwen_vl_key}"},
                    json=body,
                )
                response.raise_for_status()
                return response

        with self._request_lock:
            response = call_with_retry(
                request,
                max_retries=self.settings.remote_max_retries,
                base_delay_seconds=self.settings.remote_retry_base_delay_seconds,
            )
        payload = response.json()
        with self._counter_lock:
            self.token_usage += int(payload.get("usage", {}).get("total_tokens", 0) or 0)
        content = payload.get("choices", [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("Qwen-VL 未返回正文内容。")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("Qwen-VL 输出不是 JSON object。")
        return value

    def _prepare_image(self, path: Path) -> tuple[str, bytes]:
        """在内存中放大小图，避免改变持久化的原始 Figure 文件。"""
        original_mime = mimetypes.guess_type(path.name)[0] or "image/png"
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened)
            width, height = image.size
            scale = max(
                1.0,
                self.MIN_FIGURE_WIDTH / max(width, 1),
                self.MIN_FIGURE_HEIGHT / max(height, 1),
            )
            if scale > 1.0:
                scale = min(
                    scale,
                    self.MAX_FIGURE_DIMENSION / max(width, height, 1),
                )
                scale = max(scale, 1.0)
            target = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            upscaled = target != (width, height)
            self.last_image_preparation = {
                "original_width": width,
                "original_height": height,
                "sent_width": target[0],
                "sent_height": target[1],
                "upscaled": upscaled,
                "scale": round(scale, 3),
            }
            if not upscaled:
                return original_mime, path.read_bytes()
            resized = image.resize(target, Image.Resampling.LANCZOS)
            if resized.mode not in ("RGB", "RGBA"):
                resized = resized.convert("RGB")
            buffer = BytesIO()
            resized.save(buffer, format="PNG", optimize=True)
            return "image/png", buffer.getvalue()
