# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Session adapters — per-session-type delivery rules.

Phase 1 ships ``ConsultingAdapter``, ``ConversationAdapter``, and
``NotificationAdapter``. Phase 2 adds the multi-participant types:
``BroadcastAdapter``, ``DiscussionAdapter`` (with static / dynamic /
round-robin orderings), and ``AuctionAdapter``.
"""

from __future__ import annotations

from .auction import AuctionAdapter
from .base import AdapterResult, SessionAdapter
from .broadcast import BroadcastAdapter
from .consulting import ConsultingAdapter
from .conversation import ConversationAdapter
from .discussion import DiscussionAdapter
from .notification import NotificationAdapter

__all__ = (
    "AdapterResult",
    "AuctionAdapter",
    "BroadcastAdapter",
    "ConsultingAdapter",
    "ConversationAdapter",
    "DiscussionAdapter",
    "NotificationAdapter",
    "SessionAdapter",
)
