# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HumanClient — actor-symmetry for human participants.

Design §13.5 says "humans are first-class actors." Phase 3a ships
:class:`HumanClient` as the first non-LLM specialization of
:class:`ActorClient` so the claim gets stress-tested before Phase 6
builds the agentic tool surface on top of it.

The split:

* :class:`HumanSurface` is a Protocol. Implementations present
  envelopes to a human operator (a CLI shell, a TUI, a web form,
  an approval webhook) and return the operator's response as a
  plain string — or ``None`` to observe silently.
* :class:`HumanClient` is an :class:`ActorClient` subclass that
  wires a single ``HumanSurface`` into all six built-in session
  types via the handler registry. Identity / rule / inbox /
  transform plumbing is identical; the only difference is the
  notify dispatch.
* :class:`HumanCliSurface` is a concrete surface that reads from
  stdin and writes to stdout. It uses ``run_in_executor`` for the
  blocking ``input()`` call so the main event loop keeps turning.
* :class:`HumanScriptedSurface` is a test-only surface that yields
  a pre-declared sequence of responses. Used by the Phase 3a test
  suite to drive end-to-end sessions without real I/O.

Integration: :class:`HumanClient` is constructed by
:meth:`HubClient.register_human` which takes the same identity +
rule arguments as :meth:`HubClient.register` but wraps the result
in a :class:`HumanClient` with the surface pre-wired. The hub
never has to know the client is a human — from its perspective it
is just another actor with ``runtime_kind="human"`` on its
identity.

What the hub does know is the identity's ``runtime_kind``. The hub
writes this verbatim into ``hub/actors/{id}/identity.json`` at
registration time, and discovery / describe endpoints surface it
unchanged. Nothing in the hub branches on ``runtime_kind`` — the
field is purely informational for the clients talking to this
identity.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..envelope import EV_TEXT, Envelope
from ..identity import ActorIdentity
from ..rule import Rule
from ..session_types import SessionType
from ..transport.link import Link
from .actor_client import ActorClient

if TYPE_CHECKING:
    from ..hub import Hub
    from .hub_client import HubClient

log = logging.getLogger("autogen.beta.network.client.human")


# ---------------------------------------------------------------------------
# Surface protocol + concrete implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class HumanSurface(Protocol):
    """Pluggable UI layer for a :class:`HumanClient`.

    Implementations receive incoming envelopes and return the
    human's response as a plain string. Returning ``None`` means
    "observe this envelope but do not reply" — useful for
    broadcast / discussion participants that just listen.
    """

    async def on_envelope(self, envelope: Envelope, client: HumanClient) -> str | None:
        """Present ``envelope`` to the human and await their response."""
        ...

    async def on_close(self, client: HumanClient) -> None:
        """Called once when the client disconnects or is unregistered.

        Default implementations typically flush output or close a
        UI; scripted surfaces used in tests ignore this hook. The
        method is part of the protocol so surfaces can opt in.
        """
        ...


class HumanCliSurface:
    """Stdin/stdout terminal surface.

    Reads the human's response from ``stdin`` (blocking ``input()``
    call wrapped in :func:`asyncio.get_event_loop().run_in_executor`
    so the main event loop stays responsive). Writes a one-line
    prompt per incoming envelope to ``stdout``.

    ``prompt_format`` is a callable that turns an incoming envelope
    into the prompt string shown to the human. The default format
    is ``"[<sender_name>]: <content>\\n> "`` which is readable at a
    terminal but can be overridden for custom presentations.

    Empty responses (just pressing Enter) are treated as "no
    reply" — the surface returns ``None`` and the client skips the
    reply post. Type a non-empty response to reply.
    """

    def __init__(
        self,
        *,
        input_fn: Callable[[str], str] | None = None,
        output_fn: Callable[[str], None] | None = None,
        prompt_format: Callable[[Envelope], str] | None = None,
    ) -> None:
        self._input_fn = input_fn or (lambda prompt: input(prompt))
        self._output_fn = output_fn or (lambda msg: sys.stdout.write(msg))
        self._prompt_format = prompt_format or self._default_prompt_format

    @staticmethod
    def _default_prompt_format(envelope: Envelope) -> str:
        sender = envelope.sender_id or "unknown"
        try:
            content = envelope.content()
        except KeyError:
            content = f"<{envelope.event_type}>"
        return f"\n[{sender}]: {content}\n> "

    async def on_envelope(self, envelope: Envelope, client: HumanClient) -> str | None:
        if envelope.event_type != EV_TEXT:
            # Default cli surface only surfaces text envelopes to the
            # operator. System/adapter events are observed silently.
            return None
        prompt = self._prompt_format(envelope)
        loop = asyncio.get_event_loop()
        try:
            line = await loop.run_in_executor(None, self._input_fn, prompt)
        except (EOFError, KeyboardInterrupt):
            return None
        line = (line or "").strip()
        return line or None

    async def on_close(self, client: HumanClient) -> None:
        with __import__("contextlib").suppress(Exception):
            self._output_fn("\n[human client disconnected]\n")


class HumanScriptedSurface:
    """Test-only surface that yields a pre-declared sequence of responses.

    Each incoming envelope consumes one entry from the scripted
    list. Once the list is exhausted, subsequent envelopes return
    ``None`` (silent). Useful for driving end-to-end session tests
    without any real stdin/stdout interaction — the whole test
    becomes deterministic and fast.

    The surface records every envelope it sees in :attr:`seen` so
    tests can assert on what was shown to the operator in addition
    to what was replied.
    """

    def __init__(self, responses: list[str | None]) -> None:
        self._responses: list[str | None] = list(responses)
        self._index = 0
        self.seen: list[Envelope] = []
        self.closed = False

    async def on_envelope(self, envelope: Envelope, client: HumanClient) -> str | None:
        self.seen.append(envelope)
        if envelope.event_type != EV_TEXT:
            # Scripted surface only counts text envelopes against the
            # response list — observers never "reply" to system
            # events so the scripted index is not consumed.
            return None
        if self._index >= len(self._responses):
            return None
        response = self._responses[self._index]
        self._index += 1
        return response

    async def on_close(self, client: HumanClient) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# HumanClient
# ---------------------------------------------------------------------------


class HumanClient(ActorClient):
    """:class:`ActorClient` specialization that routes notify through a surface.

    Constructed the same way as a regular :class:`ActorClient` but
    with an additional ``surface`` argument. The ``actor`` argument
    is optional — default ``None`` is appropriate for most human
    identities (the human is the work, not a Python ``Actor``). If
    a non-``None`` actor is passed it is still held, which lets
    advanced users back a human identity with a framework-core
    Actor that runs observers / watches / HITL hooks in parallel
    with the human surface.

    Default handlers for every built-in session type route through
    ``self.surface.on_envelope``. Advanced users can still override
    per-type handlers via :meth:`on` — the surface is the default,
    not the only option.

    Disconnect / unregister calls the surface's ``on_close`` hook
    so UI surfaces can flush output, close file descriptors, or
    otherwise tear down cleanly.
    """

    def __init__(
        self,
        *,
        surface: HumanSurface,
        identity: ActorIdentity,
        rule: Rule,
        hub: Hub,
        link: Link,
        hub_client: HubClient | None = None,
        actor: Any = None,
    ) -> None:
        super().__init__(
            actor=actor,
            identity=identity,
            rule=rule,
            hub=hub,
            link=link,
            hub_client=hub_client,
        )
        self._surface = surface
        # Wire the surface into every built-in session type. Users
        # who want to override a specific type still can — the
        # dictionary allows last-write-wins overrides via `.on()`.
        for session_type in SessionType:
            self._type_handlers[session_type.value] = self._handle_via_surface

    # ------------------------------------------------------------------

    @property
    def surface(self) -> HumanSurface:
        return self._surface

    @property
    def runtime_kind(self) -> str:
        """Convenience accessor for the identity's runtime kind."""

        return getattr(self._identity, "runtime_kind", "human")

    # ------------------------------------------------------------------
    # Notify dispatch
    # ------------------------------------------------------------------

    async def _handle_via_surface(
        self, envelope: Envelope, _client: ActorClient
    ) -> None:
        """Default handler: present the envelope to the surface and reply.

        Matches the ``NotifyHandler`` signature so it can be
        registered via ``self._type_handlers[...] = ...``. The
        ``_client`` argument is the same ``HumanClient`` instance
        as ``self`` — we ignore it and use ``self`` directly so
        the surface has a typed reference.
        """

        try:
            response = await self._surface.on_envelope(envelope, self)
        except Exception:  # pragma: no cover
            log.exception(
                "human surface raised on envelope %s; skipping reply",
                envelope.envelope_id,
            )
            return

        if response is None:
            return

        if envelope.event_type != EV_TEXT:
            # The surface returned a string for a non-text envelope
            # (e.g., an auction select). We don't auto-wrap system
            # events — the surface should call ``session.send`` /
            # ``session.post`` directly for those advanced flows.
            log.debug(
                "human surface returned string for non-text event %s; ignoring",
                envelope.event_type,
            )
            return

        await self._post_text_reply(envelope, response)

    # ------------------------------------------------------------------
    # Lifecycle — on_close hook
    # ------------------------------------------------------------------

    async def disconnect(self) -> None:
        try:
            await super().disconnect()
        finally:
            try:
                await self._surface.on_close(self)
            except Exception:  # pragma: no cover
                log.warning("human surface on_close raised", exc_info=True)


def human_cli_client(
    *,
    identity: ActorIdentity,
    rule: Rule,
    hub: Hub,
    link: Link,
    hub_client: HubClient | None = None,
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] | None = None,
    prompt_format: Callable[[Envelope], str] | None = None,
) -> HumanClient:
    """Factory: ``HumanClient`` wired to a :class:`HumanCliSurface`.

    The design doc calls this shape ``HumanCliClient`` — "HumanClient
    plus HumanCliSurface." We ship it as a plain factory function
    instead of a subclass because the only behavior that differs
    from a bare :class:`HumanClient` is the surface choice, and
    keeping the class hierarchy flat makes it obvious that
    ``HumanClient`` is the load-bearing abstraction and ``CliSurface``
    is "just another surface."
    """

    surface = HumanCliSurface(
        input_fn=input_fn,
        output_fn=output_fn,
        prompt_format=prompt_format,
    )
    return HumanClient(
        surface=surface,
        identity=identity,
        rule=rule,
        hub=hub,
        link=link,
        hub_client=hub_client,
    )
