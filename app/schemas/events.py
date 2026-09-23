"""Server-sent event contracts used by the single-process run manager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunEvent:
    id: int
    event: str
    data: dict[str, Any]

    def encode(self) -> str:
        import json

        return (
            f"id: {self.id}\n"
            f"event: {self.event}\n"
            f"data: {json.dumps(self.data, ensure_ascii=False)}\n\n"
        )
