from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

from fastapi import Request
from pythonjsonlogger import jsonlogger

from src.core.config import settings

correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        extra_fields = getattr(record, "extra_fields", {})
        correlation_id = correlation_id_var.get()
        request_id = request_id_var.get()
        if correlation_id:
            extra_fields["correlation_id"] = correlation_id
        if request_id:
            extra_fields["request_id"] = request_id
        if extra_fields:
            record.extra_fields = extra_fields
        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = settings.APP_NAME
        log_record["environment"] = settings.APP_ENV
        log_record["version"] = settings.APP_VERSION
        if hasattr(record, "extra_fields"):
            log_record.update(record.extra_fields)


def setup_logging() -> None:
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.addFilter(ContextFilter())

    format_type = settings.LOG_FORMAT_TYPE.lower()
    use_json = format_type == "json" or (format_type == "auto" and settings.is_production)
    if use_json:
        formatter = CustomJsonFormatter("%(message)s")
    else:
        formatter = logging.Formatter(settings.LOG_FORMAT)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_logging_context(correlation_id: str | None = None, request_id: str | None = None) -> None:
    if correlation_id:
        correlation_id_var.set(correlation_id)
    if request_id:
        request_id_var.set(request_id)


def clear_logging_context() -> None:
    correlation_id_var.set(None)
    request_id_var.set(None)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    request: Request | None = None,
    **kwargs: Any,
) -> None:
    extra_fields = dict(kwargs)
    if request is not None:
        extra_fields["method"] = request.method
        extra_fields["path"] = str(request.url.path)
        if request.client:
            extra_fields["client_ip"] = request.client.host
    logger.log(level, message, extra={"extra_fields": extra_fields})
