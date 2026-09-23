"""Validated public contracts for persistent paper conversations."""

from typing import Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=120)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MessageCreate(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    client_message_id: str = Field(min_length=8, max_length=128)


class MemoryItemUpdate(BaseModel):
    pinned: bool | None = None
    user_note: str | None = Field(default=None, max_length=2000)
    # Public clients may forget a memory, but cannot bypass evidence/version gates
    # by manually promoting stale or unresolved data to active.
    status: Literal["forgotten"] | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
