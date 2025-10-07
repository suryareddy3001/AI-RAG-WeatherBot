import structlog
import logging
import sys
from typing import Any

"""Logging configuration module for structured logging.

This module sets up structured logging using the structlog library, integrated with Python's standard logging module.
It provides a function to configure logging with JSON output and a utility to retrieve a logger instance by name.
"""

def setup_logging() -> None:
    """Configure structured logging with JSON output.

    Sets up the standard logging module with a basic configuration and configures structlog
    to use JSON rendering, timestamps, log levels, stack information, and exception formatting.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> Any:
    """Retrieve a structured logger instance by name.

    Args:
        name: The name of the logger, typically the module name.

    Returns:
        A structlog logger instance configured for structured logging.
    """
    return structlog.get_logger(name)