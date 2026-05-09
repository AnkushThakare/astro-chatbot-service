from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import get_logger, log_with_context
from app.middleware.correlation import get_correlation_id, get_request_id

logger = get_logger(__name__)


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
            correlation_id = get_correlation_id(request)
            request_id = get_request_id(request)
            process_time = round((time.time() - start_time) * 1000, 2)
            log_with_context(
                logger,
                logging.ERROR,
                f"Unhandled exception: {exc}",
                request=request,
                correlation_id=correlation_id,
                request_id=request_id,
                process_time_ms=process_time,
                error_type=type(exc).__name__,
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "correlation_id": correlation_id,
                    "request_id": request_id,
                },
                headers={"X-Process-Time": str(process_time / 1000)},
            )

