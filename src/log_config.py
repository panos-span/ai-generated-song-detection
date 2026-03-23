"""Centralized Loguru logging setup with stdlib interception."""

import logging
import sys

from loguru import logger


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure Loguru as the sole logging backend."""
    logger.remove()
    fmt = "{time:HH:mm:ss} | {level:<7} | {name}:{function}:{line} | {message}"
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    if log_file:
        logger.add(log_file, format=fmt, level="DEBUG", rotation="10 MB", retention="30 days")

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
