from __future__ import annotations

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging import clear_logging_context, set_logging_context

CORRELATION_ID_HEADER = "X-Correlation-ID"
REQUEST_ID_HEADER = "X-Request-ID"


class CorrelationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get(CORRELATION_ID_HEADER, str(uuid.uuid4()))
        request_id = str(uuid.uuid4())

        request.state.correlation_id = correlation_id
        request.state.request_id = request_id
        set_logging_context(correlation_id=correlation_id, request_id=request_id)

        try:
            response = await call_next(request)
            response.headers[CORRELATION_ID_HEADER] = correlation_id
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            clear_logging_context()


def get_correlation_id(request: Request) -> str | None:
    return getattr(request.state, "correlation_id", None)


def get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)

