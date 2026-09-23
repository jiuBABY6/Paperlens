"""Conversation application service: identifiers, ownership and API-friendly errors."""

from __future__ import annotations

import uuid

from app.repository import PaperRepository


class ConversationService:
    def __init__(self, repository: PaperRepository) -> None:
        self.repository = repository

    def create(self, paper_id: str, title: str = "新对话") -> dict:
        return self.repository.create_conversation(str(uuid.uuid4()), paper_id, title)

    def submit(self, paper_id: str, conversation_id: str, question: str,
               client_message_id: str) -> tuple[dict, dict, bool]:
        return self.repository.create_message_and_run(
            paper_id=paper_id,
            conversation_id=conversation_id,
            question=question.strip(),
            client_message_id=client_message_id,
            message_id=str(uuid.uuid4()),
            run_id=str(uuid.uuid4()),
        )
