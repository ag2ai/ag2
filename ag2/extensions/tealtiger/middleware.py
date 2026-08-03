# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TealTiger governance middleware for AG2.

Implements MiddlewareFactory pattern: long-lived factory holds shared state
(decisions, receipts, frozen agents, cumulative cost), per-turn instances
get a reference to that shared state.

Maintainer: @nagasatish007
"""

import fnmatch
import json
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ag2.annotations import Context
    from ag2.events import BaseEvent, ToolCallEvent
    from ag2.middleware.base import ToolExecution

from ag2.events import ToolErrorEvent
from ag2.middleware import BaseMiddleware
from ag2.middleware.base import MiddlewareFactory, ToolResultType
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from .types import (
    GovernanceDecision,
    GovernanceMode,
    GovernancePolicy,
    PII_PATTERNS,
    SECRET_PATTERNS,
    TEECReceipt,
)


class _GovernanceState:
    """Shared mutable state across all per-turn middleware instances."""

    def __init__(self) -> None:
        self.decisions: list[GovernanceDecision] = []
        self.receipts: list[TEECReceipt] = []
        self.frozen_agents: set[str] = set()
        self.cumulative_cost: float = 0.0


class TealTigerMiddleware(MiddlewareFactory):
    """TealTiger governance middleware factory.

    Long-lived object that holds shared governance state. Creates per-turn
    middleware instances.

    Args:
        mode:
