"""Logging configuration using loguru."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger


class InterceptHandler(logging.Handler):
    """Intercept standard logging and route it through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding loguru level
        try:
            level: str = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logged message originated
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(
    level: str = "INFO",
    log_file: Path | str | None = None,
    fmt: str | None = None,
) -> None:
    """Configure loguru logging.

    Parameters
    ----------
    level
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file
        Optional path to a log file. If provided, logs are written there in addition to stderr.
    fmt
        Loguru format string.
    """
    # Remove default handler
    logger.remove()

    if fmt is None:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    logger.add(sys.stderr, level=level, format=fmt, colorize=True, backtrace=True, diagnose=False)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_file),
            level=level,
            format=fmt,
            rotation="50 MB",
            retention="30 days",
            compression="gz",
            backtrace=True,
            diagnose=False,
        )

    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Quiet noisy libraries
    for noisy in ("urllib3", "matplotlib", "PIL", "asyncio", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Make PyTorch Lightning use loguru too
    try:
        from pytorch_lightning import loggers as pl_loggers  # noqa: F401
    except Exception:
        pass


def get_logger(name: str = __name__):
    """Return a loguru-bound logger."""
    return logger.bind(name=name)
