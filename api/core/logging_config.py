"""
Structured JSON logging (SPEC §9).

``configure_logging`` sets up ONE JSON pipeline per process using structlog's
``ProcessorFormatter``: structlog events and foreign stdlib records (Celery,
uvicorn, SQLAlchemy) are all rendered to JSON with a consistent field set —
``timestamp, level, service, logger, message`` plus any bound context
(``job_id, worker_id, event, duration_ms``). Call once at API/worker startup.
"""

import logging
import sys
from typing import Any

import structlog

from api.config import settings


def _add_service(service_name: str):
    """Return a structlog processor that stamps every event with the service."""

    def processor(logger: Any, method_name: str, event_dict: dict) -> dict:
        event_dict["service"] = service_name
        return event_dict

    return processor


def configure_logging(service_name: str) -> None:
    """Configure structlog + stdlib logging to emit JSON. Idempotent."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Shared across structlog events and (via foreign_pre_chain) stdlib records,
    # so both carry timestamp/level/logger/service before rendering.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
        _add_service(service_name),
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Final rendering shared by structlog + foreign records.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
