# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network assembly policies.

Phase 2 ships two :class:`~autogen.beta.assembly.AssemblyPolicy`
implementations that bridge session WAL state into a framework-core
actor's LLM view:

* :class:`SessionInboxPolicy` — reads the full session WAL, converts
  every text envelope into a ``ModelRequest`` / ``ModelMessage`` pair,
  and prepends them to the actor's model events. This is the missing
  deferred piece from Phase 1; Phase 1 leaned on ``Actor.history``
  locally to preserve multi-turn context, which was fine for two-party
  sessions but leaves multi-party discussions out of the model view.
* :class:`PreviousOnlyInboxPolicy` — only injects the most recent
  prior envelope addressed to the receiver. This is the V2 *pipeline*
  topology replacement: in a ``discussion(ordering="static")`` session
  each stage only sees the previous stage's output.
"""

from __future__ import annotations

from .session_inbox import (
    ACTOR_CLIENT_DEP,
    HUB_DEP,
    PreviousOnlyInboxPolicy,
    SESSION_DEP,
    SESSION_ID_VAR,
    SessionInboxPolicy,
    TASK_DEP,
)

__all__ = (
    "ACTOR_CLIENT_DEP",
    "HUB_DEP",
    "PreviousOnlyInboxPolicy",
    "SESSION_DEP",
    "SESSION_ID_VAR",
    "SessionInboxPolicy",
    "TASK_DEP",
)
