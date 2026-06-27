# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Federation seam: ``parse_hub_urn``, ``Passport.hub_id``, and the
``RemoteAgentProxy`` registry + dispatch path on :class:`Hub`.

A test-double ``RecordingProxy`` stands in for any real federation
transport — it records each ``dispatch`` call so tests can assert the
hub handed envelopes to the proxy under the right conditions.
"""

import asyncio

import pytest

from ag2.knowledge import MemoryKnowledgeStore
from ag2.network import (
    EV_CHANNEL_INVITE,
    EV_CHANNEL_INVITE_ACK,
    EV_TEXT,
    AuthBlock,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Passport,
    RemoteAgentProxy,
    Resume,
    parse_hub_urn,
)


class _RecordingProxy:
    """Test double: satisfies :class:`RemoteAgentProxy` by recording
    every call. Lets tests inspect what the hub handed off without
    needing a real wire transport.

    When ``hub`` is supplied, invites dispatched to the recipient are
    auto-acked back through ``hub.post_envelope`` — simulating what a
    real federation bridge would do once the peer hub confirms the
    invite. Without it the proxy is record-only.
    """

    def __init__(
        self,
        scheme: str,
        *,
        hub: Hub | None = None,
        raise_on_dispatch: BaseException | None = None,
    ) -> None:
        self.scheme = scheme
        self.calls: list[tuple[Envelope, Passport]] = []
        self.closed = False
        self._hub = hub
        self._raise_on_dispatch = raise_on_dispatch

    async def dispatch(self, envelope: Envelope, recipient: Passport) -> None:
        self.calls.append((envelope, recipient))
        if self._raise_on_dispatch is not None:
            raise self._raise_on_dispatch
        if self._hub is not None and envelope.event_type == EV_CHANNEL_INVITE and recipient.agent_id is not None:
            ack = Envelope(
                channel_id=envelope.channel_id,
                sender_id=recipient.agent_id,
                audience=None,
                event_type=EV_CHANNEL_INVITE_ACK,
                event_data={"channel_id": envelope.channel_id},
                causation_id=envelope.envelope_id,
            )
            await self._hub.post_envelope(ack)

    async def close(self) -> None:
        self.closed = True


class _RecordingListener:
    """HubListener that captures on_dispatch_failed for assertion."""

    def __init__(self) -> None:
        self.dispatch_failures: list[tuple[Envelope, str, BaseException]] = []

    async def on_envelope_posted(self, envelope, metadata):
        pass

    async def on_envelope_rejected(self, envelope, reason):
        pass

    async def on_dispatch_failed(self, envelope, recipient_id, reason):
        self.dispatch_failures.append((envelope, recipient_id, reason))

    async def on_channel_event(self, channel_id, kind, payload):
        pass

    async def on_agent_event(self, agent_id, kind, payload):
        pass

    async def on_expectation_fired(self, channel_id, expectation, violation):
        pass

    async def on_turn_failed(self, channel_id, agent_id, envelope_id, exc):
        pass

    async def on_task_event(self, task_id, kind, payload):
        pass

    async def on_inbox_pressure(self, agent_id, pending, cap):
        pass


async def _new_hub() -> Hub:
    return await Hub.open(
        MemoryKnowledgeStore(),
        ttl_sweep_interval=0,
        expectation_sweep_interval=0,
    )


class TestParseHubUrn:
    def test_valid_urn_splits_into_hub_and_agent(self) -> None:
        assert parse_hub_urn("hub://hub-1/agent-42") == ("hub-1", "agent-42")

    def test_non_urn_input_returns_none_and_passthrough(self) -> None:
        assert parse_hub_urn("just-an-agent-id") == (None, "just-an-agent-id")
        assert parse_hub_urn("") == (None, "")

    def test_malformed_urns_are_treated_as_passthrough(self) -> None:
        # Missing slash after hub_id.
        assert parse_hub_urn("hub://only-hub-id") == (None, "hub://only-hub-id")
        # Empty hub_id.
        assert parse_hub_urn("hub:///agent-42") == (None, "hub:///agent-42")
        # Empty agent_id.
        assert parse_hub_urn("hub://hub-1/") == (None, "hub://hub-1/")

    def test_round_trip_is_idempotent(self) -> None:
        hub_id, agent_id = parse_hub_urn("hub://h/a")
        # Reconstruct and re-parse — should land on the same parts.
        reconstructed = f"hub://{hub_id}/{agent_id}"
        assert parse_hub_urn(reconstructed) == (hub_id, agent_id)


class TestPassportHubId:
    def test_defaults_to_none(self) -> None:
        p = Passport(name="alice")
        assert p.hub_id is None

    def test_dict_round_trip_preserves_hub_id(self) -> None:
        p = Passport(name="alice", hub_id="other-hub", kind="remote_agent")
        restored = Passport.from_dict(p.to_dict())
        assert restored.hub_id == "other-hub"
        assert restored.effective_kind == "remote_agent"


class TestRemoteProxyRegistry:
    @pytest.mark.asyncio
    async def test_register_and_lookup_by_scheme(self) -> None:
        hub = await _new_hub()
        try:
            proxy = _RecordingProxy("a2a")
            hub.register_remote_proxy(proxy)
            assert hub.remote_proxy_for("a2a") is proxy
            assert hub.remote_proxy_for("unknown") is None
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_register_replaces_prior_proxy_at_same_scheme(self) -> None:
        hub = await _new_hub()
        try:
            first = _RecordingProxy("a2a")
            second = _RecordingProxy("a2a")
            hub.register_remote_proxy(first)
            hub.register_remote_proxy(second)
            assert hub.remote_proxy_for("a2a") is second
        finally:
            await hub.close()

    @pytest.mark.asyncio
    async def test_unregister_returns_and_removes(self) -> None:
        hub = await _new_hub()
        try:
            proxy = _RecordingProxy("a2a")
            hub.register_remote_proxy(proxy)
            removed = hub.unregister_remote_proxy("a2a")
            assert removed is proxy
            assert hub.remote_proxy_for("a2a") is None
            assert hub.unregister_remote_proxy("a2a") is None
        finally:
            await hub.close()

    def test_protocol_runtime_checkable(self) -> None:
        proxy = _RecordingProxy("a2a")
        assert isinstance(proxy, RemoteAgentProxy)


class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_remote_agent_recipient_routes_through_proxy(self) -> None:
        """Alice posts a text addressed to a remote bob. The hub should
        call ``proxy.dispatch`` instead of trying to deliver locally."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)

        from ag2 import Agent

        from ._helpers import ScriptedConfig

        alice = await alice_hc.register(
            Agent(name="alice", config=ScriptedConfig()),
            Passport(name="alice"),
            Resume(),
        )

        # Register bob's passport as a remote agent (kind="remote_agent",
        # auth.scheme="a2a"). Construct the passport directly with an
        # agent_id so the hub's name index agrees.
        bob_passport = Passport(
            name="bob",
            auth=AuthBlock(scheme="a2a", claim={"endpoint": "wss://other-hub/bob"}),
            kind="remote_agent",
            hub_id="other-hub",
        )
        bob_passport = await hub.register_identity(bob_passport, Resume())

        proxy = _RecordingProxy("a2a", hub=hub)
        hub.register_remote_proxy(proxy)

        try:
            channel = await alice.open(type="conversation", target="bob")
            await channel.send("hello bob")
            await asyncio.sleep(0.05)

            # Find the EV_TEXT envelope alice posted.
            text_calls = [
                (env, rcp)
                for env, rcp in proxy.calls
                if env.event_type == EV_TEXT and env.event_data.get("text") == "hello bob"
            ]
            assert len(text_calls) == 1
            env, rcp = text_calls[0]
            assert rcp.agent_id == bob_passport.agent_id
            assert rcp.hub_id == "other-hub"
            assert rcp.effective_kind == "remote_agent"
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_missing_proxy_fires_dispatch_failed_listener(self) -> None:
        """A remote-agent recipient whose scheme has no registered proxy
        produces an ``on_dispatch_failed`` event."""
        hub = await _new_hub()
        listener = _RecordingListener()
        hub.register_listener(listener)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)

        from ag2 import Agent

        from ._helpers import ScriptedConfig

        alice = await alice_hc.register(
            Agent(name="alice", config=ScriptedConfig()),
            Passport(name="alice"),
            Resume(),
        )
        bob_passport = await hub.register_identity(
            Passport(name="bob", auth=AuthBlock(scheme="grpc", claim={}), kind="remote_agent"),
            Resume(),
        )

        # The invite leg also lands on the (missing) proxy path, so
        # ``alice.open`` would hang on the invite-ack. Pre-ack the
        # invite synthetically before sending the substantive text.
        try:
            open_task = asyncio.create_task(alice.open(type="conversation", target="bob"))
            await asyncio.sleep(0.05)
            # Pull the channel id out of the hub's in-flight metadata.
            channel_id = next(iter(hub._active_channels))
            ack = Envelope(
                channel_id=channel_id,
                sender_id=bob_passport.agent_id,
                audience=None,
                event_type=EV_CHANNEL_INVITE_ACK,
                event_data={"channel_id": channel_id},
            )
            await hub.post_envelope(ack)
            channel = await open_task

            # Discard the invite-leg failure (no proxy for grpc) and
            # assert the EV_TEXT failure separately.
            listener.dispatch_failures.clear()

            await channel.send("hello")
            await asyncio.sleep(0.05)

            text_failures = [
                (e, rid, reason) for (e, rid, reason) in listener.dispatch_failures if e.event_type == EV_TEXT
            ]
            assert len(text_failures) == 1
            _, _, reason = text_failures[0]
            assert "no remote proxy" in str(reason).lower()
            assert "grpc" in str(reason)
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_proxy_exception_fires_dispatch_failed(self) -> None:
        hub = await _new_hub()
        listener = _RecordingListener()
        hub.register_listener(listener)
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)

        from ag2 import Agent

        from ._helpers import ScriptedConfig

        alice = await alice_hc.register(
            Agent(name="alice", config=ScriptedConfig()),
            Passport(name="alice"),
            Resume(),
        )
        bob_passport = await hub.register_identity(
            Passport(name="bob", auth=AuthBlock(scheme="a2a"), kind="remote_agent"),
            Resume(),
        )

        boom = RuntimeError("transport down")
        # Proxy raises on every dispatch, so the invite cannot
        # auto-ack via the bridge — synthesise the invite-ack inline.
        proxy = _RecordingProxy("a2a", raise_on_dispatch=boom)
        hub.register_remote_proxy(proxy)

        try:
            open_task = asyncio.create_task(alice.open(type="conversation", target="bob"))
            await asyncio.sleep(0.05)
            channel_id = next(iter(hub._active_channels))
            ack = Envelope(
                channel_id=channel_id,
                sender_id=bob_passport.agent_id,
                audience=None,
                event_type=EV_CHANNEL_INVITE_ACK,
                event_data={"channel_id": channel_id},
            )
            await hub.post_envelope(ack)
            channel = await open_task
            # Drop the invite-leg failure record so we assert only the text.
            listener.dispatch_failures.clear()
            proxy.calls.clear()

            await channel.send("hello")
            await asyncio.sleep(0.05)

            text_failures = [
                (e, rid, reason) for (e, rid, reason) in listener.dispatch_failures if e.event_type == EV_TEXT
            ]
            assert len(text_failures) == 1
            assert text_failures[0][2] is boom
            # The proxy was actually called — failure happened inside it.
            assert len(proxy.calls) == 1
        finally:
            await alice_hc.close()
            await hub.close()

    @pytest.mark.asyncio
    async def test_proxy_failure_does_not_block_local_recipients(self) -> None:
        """An envelope addressed to both a local and a remote recipient
        should reach the local one even when the remote proxy raises."""
        hub = await _new_hub()
        link = LocalLink(hub)
        alice_hc = HubClient(link, hub=hub)
        carol_hc = HubClient(link, hub=hub)

        from ag2 import Agent

        from ._helpers import ScriptedConfig

        alice = await alice_hc.register(
            Agent(name="alice", config=ScriptedConfig()),
            Passport(name="alice"),
            Resume(),
        )
        carol = await carol_hc.register(
            Agent(name="carol", config=ScriptedConfig()),
            Passport(name="carol"),
            Resume(),
        )
        bob_passport = await hub.register_identity(
            Passport(name="bob", auth=AuthBlock(scheme="a2a"), kind="remote_agent"),
            Resume(),
        )

        # Proxy raises on EV_TEXT dispatch but auto-acks invites so
        # the channel can activate.
        class _SelectiveProxy:
            scheme = "a2a"

            def __init__(self) -> None:
                self.calls: list[tuple[Envelope, Passport]] = []

            async def dispatch(self, envelope: Envelope, recipient: Passport) -> None:
                self.calls.append((envelope, recipient))
                if envelope.event_type == EV_CHANNEL_INVITE and recipient.agent_id:
                    ack = Envelope(
                        channel_id=envelope.channel_id,
                        sender_id=recipient.agent_id,
                        audience=None,
                        event_type=EV_CHANNEL_INVITE_ACK,
                        event_data={"channel_id": envelope.channel_id},
                        causation_id=envelope.envelope_id,
                    )
                    await hub.post_envelope(ack)
                    return
                if envelope.event_type == EV_TEXT:
                    raise RuntimeError("nope")

            async def close(self) -> None:
                pass

        proxy = _SelectiveProxy()
        hub.register_remote_proxy(proxy)

        received: list[Envelope] = []
        orig = carol._on_envelope

        async def cap(env):
            received.append(env)
            if orig is not None:
                await orig(env)

        carol.on_envelope(cap)

        try:
            # 3-party discussion so alice can address both carol and bob
            # in one envelope.
            channel = await alice.open(type="discussion", target=["carol", "bob"])
            await channel.send("hello", audience=[carol.agent_id, bob_passport.agent_id])
            await asyncio.sleep(0.05)

            # Carol received the text in spite of the proxy failure.
            assert any(e.event_type == EV_TEXT and e.event_data.get("text") == "hello" for e in received)
            # Proxy was still attempted for bob.
            text_calls = [env for env, _ in proxy.calls if env.event_type == EV_TEXT]
            assert len(text_calls) == 1
        finally:
            await alice_hc.close()
            await carol_hc.close()
            await hub.close()
