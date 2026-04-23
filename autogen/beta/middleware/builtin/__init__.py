# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

from .history_limiter import HistoryLimiter
from .llm_retry import RetryMiddleware
from .logging import LoggingMiddleware
from .token_limiter import TokenLimiter
from .tools import approval_required

try:
    from .telemetry import TelemetryMiddleware
except ImportError as e:
    TelemetryMiddleware = missing_optional_dependency("TelemetryMiddleware", "tracing", e)

__all__ = (
    "HistoryLimiter",
    "LoggingMiddleware",
    "RetryMiddleware",
    "TelemetryMiddleware",
    "TokenLimiter",
    "approval_required",
)
