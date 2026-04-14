# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""SessionInboxPolicy — pipe the session WAL into the actor's LLM view.

The network session's WAL is the source of truth for what happened
between actors, but a session is **not** the same as an agent memory
(§5.6): the WAL holds every system and user envelope, while the LLM
only wants to see a curated slice of text messages addressed to (or
from) its actor. This policy does the bridge.

Integration model — how the policy knows the session:

* The :class:`ActorClient` notify handler sets two entries before
  calling :meth:`Actor.ask`:

  - ``variables[SESSION_ID_VAR]`` = current session id (str)
  - ``dependencies[HUB_DEP]`` = the :class:`Hub` the client is attached to

  If either is missing the policy is a no-op — the actor might be
  running standalone (no hub) or outside a network notify handler.

* At LLM-call time the policy reads the session's WAL via
  ``Hub.read_wal`` and converts each text envelope into a
  :class:`ModelRequest` (from someone else) or :class:`ModelMessage`
  (from self). The converted events are **prepended** to the model
  events list so they appear in chronological order before the
  current turn's ``ModelRequest``.

* A ``self_actor_id`` field on the policy decides whose perspective to
  use when distinguishing "my past turns" from "other actors' past
  turns". The :class:`ActorClient` fills this in at policy-attach time
  — so one policy instance per client is correct.

The policy is stateless between invocations: the WAL is the single
source of truth, and we re-read it on every apply. This mirrors the
framework-core principle that assembly policies are pure transforms
(§assembly.py module doc).

:class:`PreviousOnlyInboxPolicy` is a strict variant used by the
static-discussion / pipeline pattern: it injects only the **previous
prior envelope** (plus any system envelopes preceding it). This maps
directly onto the V2 "each stage sees only the previous stage's
output" pipeline semantic without a separate session-type adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent, ModelMessage, ModelRequest

from ..envelope import EV_TEXT, Envelope

SESSION_ID_VAR = "ag2.network.session_id"
HUB_DEP = "ag2.network.hub"


def _envelope_to_model_event(
    envelope: Envelope, *, self_actor_id: str
) -> BaseEvent | None:
    """Translate a network envelope into a model-side event.

    Only text envelopes produce LLM-visible events. System envelopes
    (invite, ack, opened, closed, auction.select, …) return ``None`` —
    they are session metadata, not conversation content.
    """

    if envelope.event_type != EV_TEXT:
        return None
    try:
        content = envelope.content()
    except KeyError:
        return None
    if envelope.sender_id == self_actor_id:
        return ModelMessage(content=content)
    return ModelRequest.ensure_request([content])


@dataclass
class SessionInboxPolicy:
    """Prepend the full session WAL onto the actor's model events.

    ``self_actor_id`` is the id this client has been registered under
    — used to decide which envelopes are "mine" (→ ``ModelMessage``)
    vs "someone else's" (→ ``ModelRequest``).

    ``include_own`` keeps the actor's own past turns in the model
    view. When false, the policy only injects events from other
    participants (useful when framework-core ``Actor.history`` is
    already carrying the local side). Defaults to ``True`` so a fresh
    Actor without local history sees the whole exchange.
    """

    self_actor_id: str
    name: str = "session_inbox"
    include_own: bool = True

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        session_id = _resolve_var(context, SESSION_ID_VAR)
        hub = _resolve_dep(context, HUB_DEP)
        if session_id is None or hub is None:
            return prompts, events

        wal = await hub.read_wal(session_id)
        translated: list[BaseEvent] = []
        for env in wal:
            if not self.include_own and env.sender_id == self.self_actor_id:
                continue
            model_event = _envelope_to_model_event(
                env, self_actor_id=self.self_actor_id
            )
            if model_event is not None:
                translated.append(model_event)

        if not translated:
            return prompts, events

        # Keep the current turn's events *after* the prior WAL so the
        # LLM sees them in chronological order: prior history, then
        # the envelope the handler is answering right now.
        return prompts, [*translated, *events]


@dataclass
class PreviousOnlyInboxPolicy:
    """Inject only the most recent text envelope from someone else.

    Direct V2 pipeline replacement: in a
    ``discussion(ordering="static")`` session, each participant's LLM
    only sees the previous stage's output, not the whole transcript.
    The framework-core actor never needs to know about the session
    chain — it's handed one ``ModelRequest`` per turn.

    Fall-back behavior: if no previous envelope from someone else
    exists in the WAL (e.g. this is the first speaker), the policy is
    a no-op and the actor sees only the current turn's events.
    """

    self_actor_id: str
    name: str = "previous_only_inbox"

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        session_id = _resolve_var(context, SESSION_ID_VAR)
        hub = _resolve_dep(context, HUB_DEP)
        if session_id is None or hub is None:
            return prompts, events

        wal = await hub.read_wal(session_id)
        previous: ModelRequest | None = None
        for env in wal:
            if env.event_type != EV_TEXT:
                continue
            if env.sender_id == self.self_actor_id:
                continue
            try:
                content = env.content()
            except KeyError:
                continue
            previous = ModelRequest.ensure_request([content])

        if previous is None:
            return prompts, events

        return prompts, [previous, *events]


# ---------------------------------------------------------------------------
# Context accessors
# ---------------------------------------------------------------------------


def _resolve_var(context: Context, key: str) -> object:
    """Best-effort lookup in ``context.variables`` for our key.

    ``Context.variables`` is typed as a mapping-like object; Phase 2
    uses string keys for our two injection points. Subclasses of
    ``Context`` may implement variables differently (e.g. keyed by a
    ``Variable`` sentinel) — we fall back to ``None`` in that case so
    the policy stays a no-op instead of raising.
    """

    variables = getattr(context, "variables", None)
    if variables is None:
        return None
    try:
        return variables.get(key)
    except AttributeError:
        try:
            return variables[key]  # type: ignore[index]
        except (KeyError, TypeError):
            return None


def _resolve_dep(context: Context, key: str) -> object:
    deps = getattr(context, "dependencies", None)
    if deps is None:
        return None
    try:
        return deps.get(key)
    except AttributeError:
        try:
            return deps[key]  # type: ignore[index]
        except (KeyError, TypeError):
            return None
