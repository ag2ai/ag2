# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end integration tests for the Phase 5a.1 transforms pipeline.

Covers the round-trip from a rule upload through the hub's
``RuleChangedFrame`` emission, the client's pipeline rebuild, and the
actual envelope running through all four stages. Unit-level dispatch
tests live in ``test_transforms_unit.py``.

Scenarios:

- stdlib named transforms observed end-to-end via ``session.ask``.
- ``set_rule`` push swaps the pipeline live and subsequent envelopes
  see the new stages.
- ``pre_send`` rejection raises :class:`TransformRejected` to the
  caller; the hub never receives the frame.
- ``pre_receive`` rejection nacks the inbound envelope cleanly.
- Unknown ``exec`` / ``ws`` forms are logged once and behave as
  pass-through.
- Multi-tenant isolation — the hub never runs ``importlib`` on
  tenant-authored ``PythonTransform`` module paths.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any

import pytest

from autogen.beta.knowledge import MemoryKnowledgeStore
from autogen.beta.network import (
    ActorIdentity,
    Envelope,
    Hub,
    HubClient,
    LocalLink,
    Rule,
    SessionType,
    TransformContext,
    TransformRejected,
    TransformSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _EchoReply:
    content: str


class _EchoActor:
    """Minimal framework-core actor surface needed by the default handlers."""

    def __init__(self, name: str, reply: str = "") -> None:
        self.name = name
        self.reply = reply
        self.questions: list[str] = []

    async def ask(self, content: str, **_kwargs: Any) -> _EchoReply:
        self.questions.append(content)
        return _EchoReply(content=self.reply or f"{self.name}:{content}")


async def _register_pair(
    *,
    alice_rule: Rule | None = None,
    bob_rule: Rule | None = None,
    alice_install_stdlib: bool = True,
    bob_install_stdlib: bool = True,
) -> tuple[Hub, HubClient, LocalLink, Any, Any, _EchoActor, _EchoActor]:
    hub = Hub(MemoryKnowledgeStore())
    link = LocalLink()
    link.on_connection(hub.connection_handler)
    hc = HubClient(hub, link)
    alice = _EchoActor("alice", reply="alice-reply")
    bob = _EchoActor("bob", reply="bob-reply")
    alice_client = await hc.register(
        alice,
        identity=ActorIdentity(name="alice"),
        rule=alice_rule,
        install_stdlib_transforms=alice_install_stdlib,
    )
    bob_client = await hc.register(
        bob,
        identity=ActorIdentity(name="bob"),
        rule=bob_rule,
        install_stdlib_transforms=bob_install_stdlib,
    )
    return hub, hc, link, alice_client, bob_client, alice, bob


async def _teardown(hc: HubClient, link: LocalLink) -> None:
    await hc.close()
    await link.close()


# ---------------------------------------------------------------------------
# Standard library named transforms — end-to-end
# ---------------------------------------------------------------------------


class TestStdlibTransformsEndToEnd:
    @pytest.mark.asyncio
    async def test_redact_pii_on_receive(self) -> None:
        """``redact_pii`` strips PII on the inbound side before the handler.

        Bob runs ``redact_pii`` at ``pre_receive``, so when Alice sends
        a message containing an email address, Bob's handler sees the
        redacted content.
        """

        bob_rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_receive",
                    apply="redact_pii",
                ),
            ],
        )
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair(bob_rule=bob_rule)
        )
        try:
            session = await alice_client.open(
                SessionType.CONVERSATION, target="bob"
            )
            await session.send("contact me at ada@example.com please")
            # Give bob's handler time to dispatch.
            for _ in range(40):
                if bob.questions:
                    break
                await asyncio.sleep(0.01)
            assert bob.questions, "bob should have received the envelope"
            # PII was redacted before the handler saw it.
            assert "ada@example.com" not in bob.questions[0]
            assert "[REDACTED:email]" in bob.questions[0]
        finally:
            await _teardown(hc, link)

    @pytest.mark.asyncio
    async def test_redact_pii_on_send(self) -> None:
        """``redact_pii`` at ``pre_send`` rewrites the envelope before it leaves.

        Alice redacts on send, so the envelope arrives at Bob already
        sanitized — the hub's WAL records the redacted content.
        """

        alice_rule = Rule(
            transforms=[
                TransformSpec(stage="pre_send", apply="redact_pii"),
            ],
        )
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair(alice_rule=alice_rule)
        )
        try:
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await session.send("call me at (415) 555-0123 now")
            for _ in range(40):
                if bob.questions:
                    break
                await asyncio.sleep(0.01)
            assert bob.questions
            assert "(415) 555-0123" not in bob.questions[0]
            assert "[REDACTED:phone]" in bob.questions[0]
        finally:
            await _teardown(hc, link)

    @pytest.mark.asyncio
    async def test_truncate_long_content_preserves_short(self) -> None:
        """``truncate_long_content`` leaves short envelopes alone."""

        bob_rule = Rule(
            transforms=[
                TransformSpec(stage="pre_receive", apply="truncate_long_content"),
            ],
        )
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair(bob_rule=bob_rule)
        )
        try:
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await session.send("short message")
            for _ in range(40):
                if bob.questions:
                    break
                await asyncio.sleep(0.01)
            assert bob.questions == ["short message"]
        finally:
            await _teardown(hc, link)

    @pytest.mark.asyncio
    async def test_stamp_audit_header_annotates_inbound(self) -> None:
        """``stamp_audit_header`` adds ``_audit`` to the envelope data.

        We observe the side effect by installing a custom handler
        that captures the envelope post-transform and asserts the
        audit block is present.
        """

        bob_rule = Rule(
            transforms=[
                TransformSpec(stage="pre_receive", apply="stamp_audit_header"),
            ],
        )
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair(bob_rule=bob_rule)
        )
        captured: list[dict] = []
        try:
            # Swap bob's notification handler to just record the envelope.
            @bob_client.on(SessionType.NOTIFICATION)
            async def _capture(envelope: Envelope, _client: Any) -> None:
                captured.append(dict(envelope.event_data))

            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await session.send("hi")
            for _ in range(40):
                if captured:
                    break
                await asyncio.sleep(0.01)
            assert captured
            audit = captured[0].get("_audit")
            assert audit is not None
            assert audit["stage"] == "pre_receive"
            assert audit["sender_id"] == alice_client.actor_id
            assert audit["actor_id"] == bob_client.actor_id
            assert audit["rule_version"] == 1
        finally:
            await _teardown(hc, link)


# ---------------------------------------------------------------------------
# Rule_changed live swap
# ---------------------------------------------------------------------------


class TestRuleChangedLiveSwap:
    @pytest.mark.asyncio
    async def test_hub_set_rule_swaps_pipeline_in_place(self) -> None:
        """A live ``set_rule`` call rebuilds the ActorClient's pipeline."""

        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair()
        )
        try:
            # Phase 1 rule on bob: empty transforms.
            original_pipeline = bob_client._pipeline
            # Push a new rule that installs a redact_pii pre_receive.
            new_rule = Rule(
                version=2,
                transforms=[
                    TransformSpec(stage="pre_receive", apply="redact_pii"),
                ],
            )
            await hub.set_rule(bob_client.actor_id, new_rule)
            # The hub pushes RuleChangedFrame asynchronously; poll.
            for _ in range(40):
                if bob_client._pipeline is not original_pipeline:
                    break
                await asyncio.sleep(0.01)
            assert bob_client._pipeline is not original_pipeline
            assert bob_client._pipeline.rule_version == 2

            # Send a PII-bearing envelope; bob should see the redacted form.
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await session.send("ping from ada@example.com")
            for _ in range(40):
                if bob.questions:
                    break
                await asyncio.sleep(0.01)
            assert bob.questions
            assert "[REDACTED:email]" in bob.questions[0]
        finally:
            await _teardown(hc, link)


# ---------------------------------------------------------------------------
# Rejection paths
# ---------------------------------------------------------------------------


class TestTransformRejection:
    @pytest.mark.asyncio
    async def test_pre_send_rejection_raises_to_caller(self) -> None:
        """A ``pre_send`` reject raises TransformRejected from session.send."""

        # Register a rejecting named transform on alice's registry via
        # a custom factory, then reference it in the rule.
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair()
        )
        try:
            async def _always_reject(
                envelope: Envelope, ctx: TransformContext
            ) -> None:
                return None

            alice_client.register_transform(
                "always_reject", lambda: _always_reject
            )
            await hub.set_rule(
                alice_client.actor_id,
                Rule(
                    version=2,
                    transforms=[
                        TransformSpec(stage="pre_send", apply="always_reject"),
                    ],
                ),
            )
            # Wait for the rebuild.
            for _ in range(40):
                if alice_client._pipeline.rule_version == 2:
                    break
                await asyncio.sleep(0.01)
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            with pytest.raises(TransformRejected):
                await session.send("should not send")
            assert bob.questions == [], "bob must not have seen the envelope"
        finally:
            await _teardown(hc, link)

    @pytest.mark.asyncio
    async def test_pre_receive_rejection_nacks_and_skips_handler(self) -> None:
        """A ``pre_receive`` reject prevents the handler from running."""

        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair()
        )
        try:
            async def _reject_text(
                envelope: Envelope, ctx: TransformContext
            ) -> None:
                return None

            bob_client.register_transform(
                "reject_text", lambda: _reject_text
            )
            # Open the session BEFORE uploading the new rule — the
            # invite handshake is a system envelope path; the new
            # rule only gates text traffic via the ``when`` filter so
            # the session invite still goes through.
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await hub.set_rule(
                bob_client.actor_id,
                Rule(
                    version=2,
                    transforms=[
                        TransformSpec(
                            stage="pre_receive",
                            apply="reject_text",
                            when={"event": "ag2.msg.text"},
                        ),
                    ],
                ),
            )
            for _ in range(40):
                if bob_client._pipeline.rule_version == 2:
                    break
                await asyncio.sleep(0.01)
            await session.send("hello bob")
            # Give the inbox loop a chance.
            await asyncio.sleep(0.05)
            assert bob.questions == [], (
                "bob's handler should never see a rejected envelope"
            )
        finally:
            await _teardown(hc, link)


# ---------------------------------------------------------------------------
# Unknown forms (forward-compat for exec / ws in 5b)
# ---------------------------------------------------------------------------


class TestUnknownForms:
    @pytest.mark.asyncio
    async def test_exec_form_passes_through_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bob_rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_receive",
                    apply={"exec": ["./policy-filter"]},
                ),
            ],
        )
        with caplog.at_level(
            logging.WARNING,
            logger="autogen.beta.network.client.transforms.pipeline",
        ):
            hub, hc, link, alice_client, bob_client, alice, bob = (
                await _register_pair(bob_rule=bob_rule)
            )
            try:
                session = await alice_client.open(
                    SessionType.NOTIFICATION, target="bob"
                )
                await session.send("delivered")
                for _ in range(40):
                    if bob.questions:
                        break
                    await asyncio.sleep(0.01)
                # Envelope went through — the unknown form behaved as
                # pass-through, not as a reject.
                assert bob.questions == ["delivered"]
            finally:
                await _teardown(hc, link)

        exec_warnings = [
            r for r in caplog.records if "form=exec" in r.getMessage()
        ]
        assert exec_warnings, "expected at least one exec pass-through warning"


# ---------------------------------------------------------------------------
# Multi-tenant isolation — hub must never import tenant modules
# ---------------------------------------------------------------------------


class TestMultiTenantIsolation:
    @pytest.mark.asyncio
    async def test_hub_never_imports_python_transform_module(self) -> None:
        """The hub process must not evaluate tenant Python transform paths.

        Design §4's core claim: ``PythonTransform`` runs inside the
        recipient's :class:`ActorClient`, never in the hub. If the
        hub did ``importlib.import_module(spec.apply['python']['module'])``
        against a tenant-uploaded dotted path, a malicious tenant could
        run arbitrary code inside the shared hub process. This test
        uploads a rule that points at a fake module name that cannot
        exist — the rule must be stored fine (the hub is a dumb
        verbatim blob for transforms at parse time), but the hub must
        not attempt to import it.

        Guard: we look for the sentinel module name in ``sys.modules``
        before and after ``set_rule`` and assert it was never loaded.
        """

        sentinel = "_tenant_never_loaded_in_hub_xyzzy"
        assert sentinel not in sys.modules
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair()
        )
        try:
            # Upload a rule that references the sentinel module. This
            # rule will fail to build on the *ActorClient* side (the
            # client will log and the pipeline rebuild will raise),
            # but the hub must accept the store write and emit
            # RuleChangedFrame without importing anything.
            tenant_rule = Rule(
                version=2,
                transforms=[
                    TransformSpec(
                        stage="pre_send",
                        apply={
                            "python": {
                                "module": sentinel,
                                "class": "Guard",
                            }
                        },
                    ),
                ],
            )
            try:
                await hub.set_rule(alice_client.actor_id, tenant_rule)
            except Exception:  # pragma: no cover
                pass
            # The hub may have processed the rule but must never have
            # imported the sentinel. Give the rule_changed dispatch
            # some time to fire and fail on the client side.
            await asyncio.sleep(0.1)
            assert sentinel not in sys.modules, (
                "hub must not import tenant transform modules"
            )
            # Belt-and-suspenders: the hub's in-memory rule cache
            # holds the raw TransformSpec but no instantiated
            # transform object.
            cached = hub._rules[alice_client.actor_id]
            assert cached.transforms[0].apply["python"]["module"] == sentinel
            assert not hasattr(cached.transforms[0], "_instance")
        finally:
            await _teardown(hc, link)


# ---------------------------------------------------------------------------
# PythonTransform end-to-end (using a test-module class)
# ---------------------------------------------------------------------------


class _StampClass:
    """Simple PythonTransform target declared at module scope so importlib
    can find it via dotted path."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __call__(
        self, envelope: Envelope, ctx: TransformContext
    ) -> Envelope:
        content = envelope.event_data.get("content", "")
        envelope.event_data = {
            **envelope.event_data,
            "content": f"[{ctx.stage.value}]{content}",
        }
        return envelope


class TestPythonTransformE2E:
    @pytest.mark.asyncio
    async def test_python_transform_mutates_via_actor_client(self) -> None:
        bob_rule = Rule(
            transforms=[
                TransformSpec(
                    stage="pre_receive",
                    apply={
                        "python": {
                            "module": __name__,
                            "class": "_StampClass",
                        }
                    },
                ),
            ],
        )
        hub, hc, link, alice_client, bob_client, alice, bob = (
            await _register_pair(bob_rule=bob_rule)
        )
        try:
            session = await alice_client.open(
                SessionType.NOTIFICATION, target="bob"
            )
            await session.send("raw")
            for _ in range(40):
                if bob.questions:
                    break
                await asyncio.sleep(0.01)
            assert bob.questions == ["[pre_receive]raw"]
        finally:
            await _teardown(hc, link)
