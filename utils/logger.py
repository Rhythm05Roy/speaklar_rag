"""Structured logging with latency tracking."""
import logging
import json
import time
import sys
from typing import Any, Optional
from pythonjsonlogger import jsonlogger
from config import settings


class LatencyTracker:
    """Context manager to track latency of operations."""

    def __init__(self, operation_name: str, logger: logging.Logger):
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.latency_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.latency_ms = (time.perf_counter() - self.start_time) * 1000
        if exc_type is not None:
            self.logger.error(
                f"{self.operation_name} failed",
                extra={
                    "operation": self.operation_name,
                    "latency_ms": self.latency_ms,
                    "error": str(exc_val),
                },
            )
        else:
            self.logger.info(
                f"{self.operation_name} completed",
                extra={
                    "operation": self.operation_name,
                    "latency_ms": round(self.latency_ms, 1),
                },
            )
        return False


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up structured JSON logging."""
    logger = logging.getLogger("speaklar")
    logger.setLevel(log_level)

    # JSON formatter for structured logging
    json_formatter = jsonlogger.JsonFormatter(
        fmt='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
    )

    # Console handler with JSON output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(json_formatter)
    logger.addHandler(console_handler)

    return logger


# Global logger instance
logger = setup_logging(settings.log_level)


def log_operation(operation_name: str) -> LatencyTracker:
    """Create a latency tracker for an operation."""
    return LatencyTracker(operation_name, logger)

