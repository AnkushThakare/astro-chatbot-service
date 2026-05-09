from __future__ import annotations

import logging
import time
import uuid
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.core.logging import (
    clear_logging_context,
    get_logger,
    log_with_context,
    set_logging_context,
)

logger = get_logger(__name__)

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


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start_time = time.time()
        try:
            response = await call_next(request)
            response.headers["X-Process-Time"] = str(round(time.time() - start_time, 4))
            path = str(request.url.path)
            if path not in {"/health", "/ready", "/api/v1/health", "/api/v1/ready"}:
                log_with_context(
                    logger,
                    logging.INFO if response.status_code < 400 else logging.WARNING,
                    "Request completed",
                    request=request,
                    status_code=response.status_code,
                    process_time_ms=round((time.time() - start_time) * 1000, 2),
                )
            return response
        except Exception as exc:
            process_time = round((time.time() - start_time) * 1000, 2)
            log_with_context(
                logger,
                logging.ERROR,
                f"Unhandled exception: {exc}",
                request=request,
                process_time_ms=process_time,
                error_type=type(exc).__name__,
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
                headers={"X-Process-Time": str(process_time / 1000)},
            )
