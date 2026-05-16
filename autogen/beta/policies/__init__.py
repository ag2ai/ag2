# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Assembly policies — composable transforms for LLM context."""

from .alert import AlertPolicy
from .conversation import ConversationPolicy
from .episodic_memory import EpisodicMemoryPolicy
from .sliding_window import SlidingWindowPolicy
from .token_budget import TokenBudgetPolicy
from .working_memory import WorkingMemoryPolicy

AVAILABLE_TOOLS_DEP = "ag2.agent.available_tools"
"""Names of all tools registered for the current turn.

Stamped into ``context.dependencies`` by :class:`autogen.beta.agent.Agent`
immediately after building ``all_schemas``. Value is a ``list[str]``.
"""

__all__ = (
    "AVAILABLE_TOOLS_DEP",
    "AlertPolicy",
    "ConversationPolicy",
    "EpisodicMemoryPolicy",
    "SlidingWindowPolicy",
    "TokenBudgetPolicy",
    "WorkingMemoryPolicy",
)
