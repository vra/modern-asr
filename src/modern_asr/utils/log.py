"""Modern, unified logging for modern-asr.

Every module gets a rich-coloured, ISO-timestamped logger via
:func:`get_logger`.  The output format is::

    2026-05-01 21:30:15 │ modern_asr.models.sensevoice │ Loading model...

Usage::

    from modern_asr.utils.log import get_logger
    logger = get_logger(__name__)
    logger.info("Model loaded")
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a logger with rich, coloured, ISO-timestamped output.

    The handler is attached only once per logger name, so calling this
    repeatedly is safe.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            show_time=False,
            console=Console(stderr=True),
        )
        formatter = logging.Formatter(
            "%(asctime)s │ %(name)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
