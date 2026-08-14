# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
# SPDX-License-Identifier: Apache-2.0

"""TealTiger deterministic governance middleware for AG2.

No external dependencies — all governance evaluation is deterministic and
runs inline with no LLM in the governance path.
"""

from ag2.extensions.tealtiger.middleware import TealTigerMiddleware
from ag2.extensions.tealtiger.types import (
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    TEECReceipt,
)

__all__ = [
    "GovernanceDecision",
    "GovernanceMode",
    "GovernancePolicy",
    "TEECReceipt",
    "TealTigerMiddleware",
]
