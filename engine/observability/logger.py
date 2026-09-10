"""
engine/observability/logger.py
-------------------------------
Structured logging setup using structlog.

Outputs JSON lines in production / plain console in development.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Configure structlog for the application.

    Parameters
    ----------
    level:
        Python log level string ("DEBUG", "INFO", etc.).
    fmt:
        "json" for JSON-lines output, "console" for human-readable dev output.
    """
    try:
        import structlog

        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
        ]

        if fmt == "json":
            renderer = structlog.processors.JSONRenderer()
        else:
            renderer = structlog.dev.ConsoleRenderer(colors=True)

        structlog.configure(
            processors=shared_processors + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, level.upper(), logging.INFO)
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(level.upper())

    except ImportError:
        # Fallback if structlog not installed
        logging.basicConfig(
            level=level.upper(),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            stream=sys.stdout,
        )


def bind_context(**kwargs: Any) -> None:
    """Bind key-value pairs to the structlog context for the current coroutine/thread."""
    try:
        import structlog
        structlog.contextvars.bind_contextvars(**kwargs)
    except ImportError:
        pass


def clear_context() -> None:
    """Clear the structlog context."""
    try:
        import structlog
        structlog.contextvars.clear_contextvars()
    except ImportError:
        pass
