from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import Settings
from astro_chatbot_service.models.schemas import ChatRequest, ChatResponse
from astro_chatbot_service.services.astrology import AstrologyService
from astro_chatbot_service.services.groq import GroqClient
from astro_chatbot_service.services.memory import MemoryService
from astro_chatbot_service.services.prompt_builder import build_messages
from astro_chatbot_service.services.rag import RAGService


class ChatOrchestrator:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.settings = settings
        self.memory_service = MemoryService(db)
        self.rag_service = RAGService(db)
        self.groq_client = GroqClient(settings)
        self.astrology_service = AstrologyService(settings)

    async def generate_reply(self, request: ChatRequest) -> ChatResponse:
        memory_turns = self.memory_service.recent_for_prompt(
            request.session_id,
            self.settings.MEMORY_WINDOW_SIZE,
        )
        retrieved_docs = self.rag_service.retrieve(request.message, self.settings.RAG_TOP_K)
        astrology_summary = None
        if request.include_astrology_context:
            astrology_summary = await self.astrology_service.build_summary(request.birth_details)

        messages = build_messages(
            system_prompt=self.settings.DEFAULT_SYSTEM_PROMPT,
            memory_turns=memory_turns,
            retrieved_docs=retrieved_docs,
            user_message=request.message,
            astrology_summary=astrology_summary,
        )
        reply = await self.groq_client.generate(messages)

        self.memory_service.add_turn(request.session_id, "user", request.message)
        self.memory_service.add_turn(request.session_id, "assistant", reply)

        return ChatResponse(
            session_id=request.session_id,
            reply=reply,
            model=self.settings.GROQ_MODEL,
            memory_turns_used=len(memory_turns),
            retrieved_document_ids=[doc.id for doc in retrieved_docs],
            astrology_summary=astrology_summary,
        )
