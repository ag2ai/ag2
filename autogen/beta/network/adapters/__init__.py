# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session adapters — code half of the manifest/adapter split.

Adapters are stateless and pure. Every decision derives from
``(SessionMetadata, AdapterState)`` where ``AdapterState`` is folded
from the WAL. The hub caches the latest state per session in memory
and reconstructs it from disk on ``hydrate()`` by re-folding —
``validate_send`` and ``on_accepted`` are O(1), not O(WAL).

Built-ins: ``ConsultingAdapter`` (1Q1R), ``ConversationAdapter`` (1+1
bidirectional), ``DiscussionAdapter`` (multi-party round-robin), and
``WorkflowAdapter`` (transition-graph orchestration).
"""

from .base import AdapterResult, AdapterState, SessionAdapter
from .consulting import CONSULTING_TYPE, ConsultingAdapter, ConsultingState
from .conversation import CONVERSATION_TYPE, ConversationAdapter, ConversationState
from .discussion import (
    DISCUSSION_TYPE,
    ORDERING_ROUND_ROBIN,
    DiscussionAdapter,
    DiscussionState,
)
from .workflow import WORKFLOW_TYPE, WorkflowAdapter, WorkflowState

__all__ = (
    "CONSULTING_TYPE",
    "CONVERSATION_TYPE",
    "DISCUSSION_TYPE",
    "ORDERING_ROUND_ROBIN",
    "WORKFLOW_TYPE",
    "AdapterResult",
    "AdapterState",
    "ConsultingAdapter",
    "ConsultingState",
    "ConversationAdapter",
    "ConversationState",
    "DiscussionAdapter",
    "DiscussionState",
    "SessionAdapter",
    "WorkflowAdapter",
    "WorkflowState",
)
