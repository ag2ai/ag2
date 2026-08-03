# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TealTiger deterministic governance middleware for AG2.

Maintainer: @nagasatish007

Provides tool-call authorization, PII/secret scanning, cost budgets,
kill switches, and structured audit evidence — all deterministically
in <5ms with no LLM in the governance path.

See: https://github.com/agentguard-ai/tealtiger
"""

from ag2.exceptions import missing_additional_dependency

from .types import (
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    TEECReceipt,
)

try:
    from ._engine import create_default_engine
except ImportError as e:
    create_default_engine = missing_additional_dependency(
        "create_default_engine",
        'tealtiger>=1.3.0,<2',
        e,
    )

from .middleware import TealTigerMiddleware

__all__ = [
    "TealTigerMiddleware",
    "GovernanceDecision",
    "GovernanceMode",
    "GovernancePolicy",
    "TEECReceipt",
    "create_default_engine",
]
