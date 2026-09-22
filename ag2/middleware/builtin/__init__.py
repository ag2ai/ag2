# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2.exceptions import missing_optional_dependency

from .history_limiter import HistoryLimiter
from .llm_retry import RetryMiddleware
from .logging import LoggingMiddleware
from .token_limiter import TokenLimiter
from .tools import ApprovalRequired, approval_required

# The missing-dependency fallback rebinds a name mypy has already bound to a
# class, which it rejects outright. Showing it only the real import gives the
# names their true types; the fallback is runtime-only, exactly as it reads.
if TYPE_CHECKING:
    from .metrics import MetricsMiddleware
    from .telemetry import TelemetryMiddleware
else:
    try:
        from .telemetry import TelemetryMiddleware
    except ImportError as e:
        TelemetryMiddleware = missing_optional_dependency("TelemetryMiddleware", "tracing", e)

    try:
        from .metrics import MetricsMiddleware
    except ImportError as e:
        MetricsMiddleware = missing_optional_dependency("MetricsMiddleware", "metrics", e)

__all__ = (
    "ApprovalRequired",
    "HistoryLimiter",
    "LoggingMiddleware",
    "MetricsMiddleware",
    "RetryMiddleware",
    "TelemetryMiddleware",
    "TokenLimiter",
    "approval_required",
)
