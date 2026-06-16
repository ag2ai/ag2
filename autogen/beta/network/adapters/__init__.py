# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Channel adapters — code half of the manifest/adapter split.

Adapters are stateless and pure. Every decision derives from
``(ChannelMetadata, AdapterState)`` where ``AdapterState`` is folded
from the WAL. The hub caches the latest state per channel in memory
and reconstructs it from disk on ``hydrate()`` by re-folding —
``validate_send`` and ``on_accepted`` are O(1), not O(WAL).

Built-ins: ``ConsultingAdapter`` (1Q1R), ``ConversationAdapter`` (1+1
bidirectional), ``DiscussionAdapter`` (multi-party round-robin), and
``WorkflowAdapter`` (transition-graph orchestration).
"""

from .base import (
    AdapterResult,
    AdapterState,
    ChannelAdapter,
    ExpectedTurn,
    default_build_packet_envelope,
    default_build_round_envelope,
    default_build_text_envelope,
    default_expected_next,
    default_extract_turn_input,
    default_render_envelope,
    default_tools_for,
)
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
    "ChannelAdapter",
    "ConsultingAdapter",
    "ConsultingState",
    "ConversationAdapter",
    "ConversationState",
    "DiscussionAdapter",
    "DiscussionState",
    "ExpectedTurn",
    "WorkflowAdapter",
    "WorkflowState",
    "default_build_packet_envelope",
    "default_build_round_envelope",
    "default_build_text_envelope",
    "default_expected_next",
    "default_extract_turn_input",
    "default_render_envelope",
    "default_tools_for",
)
