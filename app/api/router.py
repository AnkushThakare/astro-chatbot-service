from fastapi import APIRouter

from app.api.endpoints import chat, fine_tuning, health, memory, retrieval

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(memory.router, tags=["memory"])
api_router.include_router(retrieval.router, tags=["retrieval"])
api_router.include_router(fine_tuning.router, tags=["fine-tuning"])

