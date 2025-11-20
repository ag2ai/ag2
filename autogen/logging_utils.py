# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from collections.abc import Iterable
from typing import Any

__all__ = ["EventStreamHandler", "event_print", "get_event_logger"]

_EVENT_LOGGER_NAME = "ag2.event.processor"
_END_KEY = "ag2_event_end"
_FLUSH_KEY = "ag2_event_flush"


class EventStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            end = getattr(record, _END_KEY, "\n")
            stream = self.stream
            stream.write(msg)
            stream.write(end)
            if getattr(record, _FLUSH_KEY, True):
                self.flush()
        except Exception:
            self.handleError(record)


def get_event_logger() -> logging.Logger:
    logger = logging.getLogger(_EVENT_LOGGER_NAME)
    if not logger.handlers:
        handler = EventStreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _stringify(objects: Iterable[Any], sep: str) -> str:
    return sep.join(str(obj) for obj in objects)


def event_print(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    flush: bool = True,
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
) -> None:
    logger = logger or get_event_logger()
    message = _stringify(objects, sep)
    extra = {_END_KEY: end, _FLUSH_KEY: flush}
    logger.log(level, message, extra=extra)
