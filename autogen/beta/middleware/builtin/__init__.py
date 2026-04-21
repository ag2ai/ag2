# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from .history_limiter import HistoryLimiter
from .llm_retry import RetryMiddleware
from .logging import LoggingMiddleware
from .token_limiter import TokenLimiter
from .tools import approval_required


def _missing_optional_dependency(name: str, extra: str, error: ImportError) -> Mock:
    def _raise_helpful_import_error(*args: object, **kwargs: object) -> None:
        raise ImportError(
            f'{name} requires optional dependencies. Install with `pip install "ag2[{extra}]"`'
        ) from error

    return Mock(side_effect=_raise_helpful_import_error)


try:
    from .telemetry import TelemetryMiddleware
except ImportError as e:
    TelemetryMiddleware = _missing_optional_dependency("TelemetryMiddleware", "tracing", e)

__all__ = (
    "HistoryLimiter",
    "LoggingMiddleware",
    "RetryMiddleware",
    "TelemetryMiddleware",
    "TokenLimiter",
    "approval_required",
)
