# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import inspect
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

__all__ = ("EventStreamHandler", "event_print", "get_current_ts", "get_event_logger", "to_dict")

# ---------------------------------------------------------------------------
# Shared sensitive-key definitions (single source of truth for all loggers)
# ---------------------------------------------------------------------------

# Exact key names treated as sensitive. Uses exact matching only — no substring
# matching — so adding a key here won't accidentally catch unrelated keys.
SENSITIVE_KEYS = frozenset({
    "api_key",
    "api-key",
    "apikey",
    "password",
    "secret",
    "credential",
    "authorization",
    "bearer",
    "access_token",
    "refresh_token",
    "auth_token",
    "api_token",
    "azure_ad_token",
    "azure_ad_token_provider",
    "azure_endpoint",
})


def get_sensitive_exclude_keys() -> tuple[str, ...]:
    """Return a tuple of sensitive key names for use with to_dict(exclude=...).

    Includes 'self' and '__class__' which SqliteLogger always excludes.
    """
    return ("self", "__class__", *SENSITIVE_KEYS)


def redact(data: Any, depth: int = 10) -> Any:
    """Recursively mask sensitive keys with '***REDACTED***'. Depth-limited to avoid cycles."""
    if depth <= 0:
        return data
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(k, str) and k.lower() in SENSITIVE_KEYS:
                out[k] = "***REDACTED***"
            else:
                out[k] = redact(v, depth - 1)
        return out
    if isinstance(data, (list, tuple, set)):
        result = [redact(item, depth - 1) for item in data]
        return type(data)(result)
    return data


_EVENT_LOGGER_NAME = "ag2.event.processor"
_END_KEY = "ag2_event_end"
_FLUSH_KEY = "ag2_event_flush"


def get_current_ts() -> str:
    """Get current timestamp in UTC timezone.

    Returns:
        str: Current timestamp in UTC timezone
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def to_dict(
    obj: int | float | str | bool | dict[Any, Any] | list[Any] | tuple[Any, ...] | Any,
    exclude: tuple[str, ...] = (),
    no_recursive: tuple[Any, ...] = (),
) -> Any:
    """Convert object to dictionary.

    Args:
        obj (Union[int, float, str, bool, dict[Any, Any], list[Any], tuple[Any, ...], Any]): Object to convert
        exclude (tuple[str, ...], optional): Keys to exclude. Defaults to ().
        no_recursive (tuple[Any, ...], optional): Types to exclude from recursive conversion. Defaults to ().
    """
    if isinstance(obj, (int, float, str, bool)):
        return obj
    elif isinstance(obj, (Path, PurePath)):
        return str(obj)
    elif callable(obj):
        return inspect.getsource(obj).strip()
    elif isinstance(obj, dict):
        return {
            str(k): to_dict(str(v)) if isinstance(v, no_recursive) else to_dict(v, exclude, no_recursive)
            for k, v in obj.items()
            if k not in exclude
        }
    elif isinstance(obj, (list, tuple)):
        return [to_dict(str(v)) if isinstance(v, no_recursive) else to_dict(v, exclude, no_recursive) for v in obj]
    elif hasattr(obj, "__dict__"):
        return {
            str(k): to_dict(str(v)) if isinstance(v, no_recursive) else to_dict(v, exclude, no_recursive)
            for k, v in vars(obj).items()
            if k not in exclude
        }
    else:
        return obj


class EventStreamHandler(logging.StreamHandler):  # type: ignore [type-arg]
    def __init__(self, stream=None):  # type: ignore [no-untyped-def]
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
        handler = EventStreamHandler(sys.stdout)  # type: ignore [no-untyped-call]
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
