# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Phase 6 — session verb unit tests (open_session, say, listen, leave)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from autogen.beta.context import ConversationContext
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ACTOR_CLIENT_DEP,
    HUB_DEP,
    SESSION_DEP,
    AccessBlock,
    ActorIdentity,
    HubClient,
    LocalLink,
    Rule,
    SessionType,
)
from autogen.beta.network.client.session import Session
from autogen.beta.network.client.verbs.session import (
    _envelope_summary,
    build_session_verbs,
)
from autogen.beta.network.envelope import Envelope
from autogen.beta.network.hub import Hub


@pytest_asyncio.fixture
async def session_setup():
    """Two clients on one hub plus an active alice→bob consulting session."""

    store = MemoryKnowledgeStore()
    hub = Hub(store)
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hub_client = HubClient(hub, link)

    class _SilentActor:
        async def ask(self, content: str, **kwargs: Any) -> str:
            # Don't reply — we want the WAL to capture only what we
            # explicitly post in the test, not auto-replies.
            return ""

    alice_client = await hub_client.register(
        _SilentActor(),
        identity=ActorIdentity(name="alice", capabilities=["ask"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )
    bob_client = await hub_client.register(
        _SilentActor(),
        identity=ActorIdentity(name="bob", capabilities=["answer"]),
        rule=Rule(access=AccessBlock(outbound_to=["*"])),
    )

    session = await alice_client.open(
        SessionType.CONVERSATION, target="bob", intent="test"
    )

    try:
        yield hub, alice_client, bob_client, session
    finally:
        await hub_client.close()
        await link.close()


def _ctx_with_session(session: Session) -> ConversationContext:
    """Build a context populated with SESSION_DEP (the verbs' fallback)."""

    return ConversationContext(
        stream=MagicMock(),
        dependencies={SESSION_DEP: session},
    )


def _ctx_empty() -> ConversationContext:
    return ConversationContext(stream=MagicMock())


async def _call(tool: Any, ctx: ConversationContext, **arguments: Any) -> Any:
    event = ToolCallEvent(
        name=tool.schema.function.name,
        arguments=json.dumps(arguments),
    )
    result: ToolResultEvent = await tool(event, ctx)
    return result.result.content


def _verbs(client: Any) -> dict[str, Any]:
    return {
        v.schema.function.name: v for v in build_session_verbs(client)
    }


@pytest.mark.asyncio
class TestOpenSession:
    async def test_returns_session_id_and_state(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob_client, _orig = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(
            verbs["open_session"],
            ctx,
            session_type="consulting",
            target="bob",
            intent="phase6 verb test",
        )
        assert "session_id" in result
        assert result["type"] == "consulting"
        assert result["state"] in {"active", "pending"}

    async def test_intent_lands_in_session_labels(
        self, session_setup
    ) -> None:
        hub, alice_client, _bob, _orig = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(
            verbs["open_session"],
            ctx,
            session_type="consulting",
            target="bob",
            intent="climate report",
        )
        meta = hub.peek_session(result["session_id"])
        assert meta is not None
        assert meta.labels.get("intent") == "climate report"

    async def test_open_stashes_new_session_in_context(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, _orig = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(
            verbs["open_session"],
            ctx,
            session_type="consulting",
            target="bob",
        )
        # After open the new session should be the new "current"
        assert SESSION_DEP in ctx.dependencies
        assert ctx.dependencies[SESSION_DEP].session_id == result["session_id"]

    async def test_unknown_target_returns_error(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, _orig = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(
            verbs["open_session"],
            ctx,
            session_type="consulting",
            target="ghost",
        )
        assert "error" in result


@pytest.mark.asyncio
class TestSay:
    async def test_say_with_default_session(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        result = await _call(verbs["say"], ctx, content="hello there")
        assert "envelope_id" in result
        assert result["session_id"] == session.session_id

    async def test_say_with_explicit_session_id(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()  # no current session in context

        result = await _call(
            verbs["say"],
            ctx,
            content="hi",
            session_id=session.session_id,
        )
        assert "envelope_id" in result

    async def test_say_without_any_session_returns_error(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, _session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(verbs["say"], ctx, content="orphan")
        assert "error" in result

    async def test_say_unknown_session_id_returns_error(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, _session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(
            verbs["say"],
            ctx,
            content="bad",
            session_id="01000000-0000-0000-0000-000000000000",
        )
        assert "error" in result


@pytest.mark.asyncio
class TestListen:
    async def test_listen_session_returns_envelope_summaries(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        # Post one message so the WAL has content beyond the
        # session-opened system envelope.
        await _call(verbs["say"], ctx, content="message 1")
        result = await _call(verbs["listen"], ctx, scope="session")

        assert result["scope"] == "session"
        assert result["session_id"] == session.session_id
        # The WAL should contain at least the system open envelope +
        # our message.
        assert len(result["envelopes"]) >= 1
        # Our message should be in there as an ag2.msg.text
        text_envs = [
            e for e in result["envelopes"] if e["event_type"] == "ag2.msg.text"
        ]
        assert any(e.get("content") == "message 1" for e in text_envs)

    async def test_listen_session_respects_limit(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        for i in range(5):
            await _call(verbs["say"], ctx, content=f"msg {i}")
        result = await _call(verbs["listen"], ctx, scope="session", limit=2)
        assert len(result["envelopes"]) == 2

    async def test_listen_unknown_scope_returns_error(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        result = await _call(verbs["listen"], ctx, scope="weird")
        assert "error" in result
        assert "scope" in result["error"]

    async def test_listen_inbox_returns_pending(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob_client, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        # bob has at least the session-invite envelope sitting in his
        # inbox after the session opened — that's what we'll see when
        # alice peeks her own inbox-equivalent. To exercise alice's
        # own inbox we need bob to send something.
        bob_verbs = _verbs(_bob_client)
        bob_ctx = ConversationContext(
            stream=MagicMock(),
            dependencies={
                SESSION_DEP: Session(
                    client=_bob_client, metadata=session.metadata
                )
            },
        )
        await _call(bob_verbs["say"], bob_ctx, content="ping from bob")

        result = await _call(verbs["listen"], ctx, scope="inbox")
        assert result["scope"] == "inbox"
        # Some envelopes should be pending in alice's inbox until her
        # default handler processes them — but the silent actor never
        # acks via the default handler so they pile up.
        assert isinstance(result["envelopes"], list)


@pytest.mark.asyncio
class TestLeave:
    async def test_leave_closes_default_session(
        self, session_setup
    ) -> None:
        hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_with_session(session)

        result = await _call(verbs["leave"], ctx)
        assert result == {
            "session_id": session.session_id,
            "state": "closed",
        }
        meta = hub.peek_session(session.session_id)
        # Closed sessions stay in the cache until archival
        assert meta is not None
        assert meta.state.value == "closed"

    async def test_leave_with_explicit_session_id(
        self, session_setup
    ) -> None:
        hub, alice_client, _bob, session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()  # no default

        result = await _call(
            verbs["leave"], ctx, session_id=session.session_id
        )
        assert result["state"] == "closed"

    async def test_leave_without_session_returns_error(
        self, session_setup
    ) -> None:
        _hub, alice_client, _bob, _session = session_setup
        verbs = _verbs(alice_client)
        ctx = _ctx_empty()

        result = await _call(verbs["leave"], ctx)
        assert "error" in result


class TestEnvelopeSummary:
    """Pin the envelope-summary shape — the LLM relies on these keys."""

    def test_text_envelope_includes_content(self) -> None:
        env = Envelope.text(
            session_id="s-1",
            sender_id="alice",
            content="hello",
            recipient_id="bob",
        )
        env.envelope_id = "e-1"
        env.created_at = "2026-04-14T00:00:00Z"
        summary = _envelope_summary(env)
        assert summary["content"] == "hello"
        assert summary["event_type"] == "ag2.msg.text"
        assert summary["sender_id"] == "alice"

    def test_non_text_envelope_includes_event_data(self) -> None:
        env = Envelope(
            session_id="s-1",
            sender_id="hub",
            recipient_id=None,
            event_type="ag2.session.opened",
            event_data={"session_id": "s-1"},
        )
        summary = _envelope_summary(env)
        assert "content" not in summary
        assert summary["event_data"] == {"session_id": "s-1"}
