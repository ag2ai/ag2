# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Default notify handlers per session type.

These handlers run inside an :class:`~.actor_client.ActorClient` when a
notify frame arrives. They perform the minimal Phase 1 behavior:

* ``consulting`` — call ``actor.ask(text)`` once, post the reply with
  ``causation_id`` set to the incoming envelope's id. The session closes
  automatically after the reply (enforced by the adapter).
* ``conversation`` — call ``actor.ask(text)`` per turn and reply with the
  result. Does not close the session — the caller is responsible.
* ``notification`` — call ``actor.ask(text)`` but do not reply. The session
  is closed by the adapter as soon as the original text lands in the WAL.

Advanced users override these via :meth:`ActorClient.on(session_type)`.

Every handler call surfaces the current session id into the actor's
``Context.variables`` under :data:`SESSION_ID_VAR` and the hub into
``Context.dependencies`` under :data:`HUB_DEP`. This is how
:class:`SessionInboxPolicy` — any assembly policy that cares about the
ambient session — finds its state without tight coupling to
``ActorClient``. Standalone ``Actor`` runs (no hub) simply don't see
these entries and continue to behave as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..envelope import EV_TEXT, Envelope
from ..policies import HUB_DEP, SESSION_ID_VAR

if TYPE_CHECKING:
    from .actor_client import ActorClient
    from .task import Task


log = logging.getLogger("autogen.beta.network.client.handlers")


async def handle_consulting(envelope: Envelope, client: ActorClient) -> None:
    if envelope.event_type != EV_TEXT:
        return
    content = envelope.content()
    reply = await _ask(client, content, envelope.session_id)
    if reply is None:
        return
    await client._post_text_reply(envelope, reply)


async def handle_conversation(envelope: Envelope, client: ActorClient) -> None:
    if envelope.event_type != EV_TEXT:
        return
    content = envelope.content()
    reply = await _ask(client, content, envelope.session_id)
    if reply is None:
        return
    await client._post_text_reply(envelope, reply)


async def handle_notification(envelope: Envelope, client: ActorClient) -> None:
    if envelope.event_type != EV_TEXT:
        return
    try:
        await _ask(client, envelope.content(), envelope.session_id)
    except Exception:  # pragma: no cover
        log.warning("notification handler: actor.ask raised", exc_info=True)


async def handle_task_assigned(
    envelope: Envelope, task: Task, client: ActorClient
) -> None:
    """Default Phase 4 task handler — bridge a framework-core ``Actor.ask``.

    The minimal useful behavior: call ``actor.ask`` with the task's
    description (falling back to the title), post the result through the
    :class:`Task` handle's :meth:`Task.result`, and let the hub drive the
    transition to ``completed``. Any exception is turned into a
    :meth:`Task.fail` so the requester's ``task.wait()`` resolves cleanly.

    Handlers that want multi-phase progress reporting, explicit partial
    results, or structured payloads override via
    :meth:`ActorClient.on_task("my-spec-type")`.
    """

    spec = task.metadata.spec
    prompt = spec.description or spec.title or ""
    try:
        answer = await _ask(client, prompt, envelope.session_id)
    except Exception as exc:
        await task.fail(f"{type(exc).__name__}: {exc}")
        return
    await task.result(answer if answer is not None else "")


async def _ask(client: ActorClient, content: str, session_id: str | None) -> str | None:
    actor = client.actor
    if actor is None:
        return f"echo: {content}"
    kwargs: dict[str, Any] = {}
    if session_id is not None:
        kwargs["variables"] = {SESSION_ID_VAR: session_id}
        kwargs["dependencies"] = {HUB_DEP: client._hub}
    try:
        reply = await actor.ask(content, **kwargs)
    except TypeError:
        # The actor's ``ask`` may not accept variables/dependencies
        # (e.g. user-defined test doubles) — fall back to the plain
        # single-argument call so Phase 1 behaviour still works.
        reply = await actor.ask(content)
    if reply is None:
        return None
    if hasattr(reply, "body"):
        return str(reply.body)
    if hasattr(reply, "content"):
        raw = reply.content
        # AgentReply.content is an async method; .body is the plain-text path.
        if callable(raw):
            return None
        return str(raw)
    return str(reply)
