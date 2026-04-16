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

Every handler call surfaces the live network state into the actor's
``Context`` before invoking ``actor.ask``:

* ``variables[SESSION_ID_VAR]``      — current session id (str)
* ``dependencies[HUB_DEP]``          — the :class:`Hub` the client is on
* ``dependencies[SESSION_DEP]``      — the live :class:`Session` handle
* ``dependencies[ACTOR_CLIENT_DEP]`` — the owning :class:`ActorClient`
* ``dependencies[TASK_DEP]``         — the in-flight :class:`Task` (task
  handler dispatch only)

This is how :class:`SessionInboxPolicy`, the auto-injected Phase 6
network verbs in :mod:`autogen.beta.network.client.verbs`, and any
user-defined tool that opts into a pre-canned alias from
:mod:`autogen.beta.network.client.inject` find their context without
tight coupling to ``ActorClient``. Standalone ``Actor`` runs (no hub)
simply don't see these entries and continue to behave as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..envelope import EV_TEXT, Envelope
from ..policies import (
    ACTOR_CLIENT_DEP,
    HUB_DEP,
    SESSION_DEP,
    SESSION_ID_VAR,
    TASK_DEP,
)

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

    The Phase 6 DI surface stamps the live :class:`Task` handle into
    ``context.dependencies[TASK_DEP]`` so the auto-injected ``track_task``
    verb (and any user tool that opts into ``TaskInject``) can resolve
    the in-flight task without explicit threading.

    Handlers that want multi-phase progress reporting, explicit partial
    results, or structured payloads override via
    :meth:`ActorClient.on_task("my-spec-type")`.
    """

    spec = task.metadata.spec
    prompt = spec.description or spec.title or ""
    try:
        answer = await _ask(client, prompt, envelope.session_id, task=task)
    except Exception as exc:
        await task.fail(f"{type(exc).__name__}: {exc}")
        return
    await task.result(answer if answer is not None else "")


async def _ask(
    client: ActorClient,
    content: str,
    session_id: str | None,
    *,
    task: Task | None = None,
) -> str | None:
    """Invoke ``actor.ask`` with full Phase 6 DI populated.

    Builds the ``Session`` handle from ``session_id`` once and stamps
    every live network handle into ``context.dependencies`` so both
    framework-injected verbs and user-defined tools can resolve them
    via the pre-canned aliases in
    :mod:`autogen.beta.network.client.inject`. Falls back to a plain
    single-argument ``actor.ask(content)`` if the underlying object
    rejects the kwargs (e.g. a user-supplied test double) so Phase 1
    behaviour stays intact.
    """

    actor = client.actor
    if actor is None:
        return f"echo: {content}"

    session = client._session_for(session_id) if session_id is not None else None

    kwargs: dict[str, Any] = {}
    if session_id is not None:
        kwargs["variables"] = {SESSION_ID_VAR: session_id}
        deps: dict[Any, Any] = {
            HUB_DEP: client._hub,
            ACTOR_CLIENT_DEP: client,
        }
        if session is not None:
            deps[SESSION_DEP] = session
        if task is not None:
            deps[TASK_DEP] = task
        kwargs["dependencies"] = deps

    # Phase 6 — auto-inject the network verb tools when the actor is
    # dispatching a real notify-handler turn. The verb factory returns
    # an empty list outside a network context, so standalone Actor
    # runs see no extra tools.
    network_verbs = client._build_network_verbs(session=session, task=task)
    if network_verbs:
        kwargs["tools"] = network_verbs

    try:
        reply = await actor.ask(content, **kwargs)
    except TypeError:
        # The actor's ``ask`` may not accept variables/dependencies/tools
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
