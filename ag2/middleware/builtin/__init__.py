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

# The fallback rebinds a name mypy has bound to a class, which it rejects; it
# sees only the real import. See website/docs/contributor-guide/type-checking.mdx.
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
