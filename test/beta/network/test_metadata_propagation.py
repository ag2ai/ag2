# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests: Envelope metadata → agent variables propagation."""

import pytest

from autogen.beta.network.convenience import Network
from autogen.beta.network.hub import Hub
from autogen.beta.network.primitives.envelope import Envelope

try:
    import aiohttp  # noqa: F401

    _has_aiohttp = True
except ImportError:
    _has_aiohttp = False

_skip_no_aiohttp = pytest.mark.skipif(not _has_aiohttp, reason="aiohttp not installed")

# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------


class _VariableCaptureAgent:
    """Agent that captures variables passed to ask()."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.received_variables: dict | None = None

    async def ask(self, message: str, **kwargs):
        self.received_variables = kwargs.get("variables")
        return type("Reply", (), {"content": self._result, "body": self._result})()


class _DelegatingAgent:
    """Agent that delegates to a sub-agent via hub._delegate()."""

    def __init__(self, name: str, hub: Hub, delegate_to: str, delegate_message: str = "do work"):
        self.name = name
        self._hub = hub
        self._delegate_to = delegate_to
        self._delegate_message = delegate_message
        self.received_variables: dict | None = None

    async def ask(self, message: str, **kwargs):
        self.received_variables = kwargs.get("variables")
        metadata = dict(self.received_variables) if self.received_variables else None
        result = await self._hub._delegate(
            self._delegate_to,
            self._delegate_message,
            source=self.name,
            metadata=metadata,
        )
        return type("Reply", (), {"content": result, "body": result})()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetadataDisabled:
    """Default Hub does NOT propagate metadata."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self) -> None:
        hub = Hub()
        agent = _VariableCaptureAgent("worker")
        await hub.register(agent)

        await hub.ask(agent, "do task", metadata={"project_id": "abc"})
        assert agent.received_variables is None

    @pytest.mark.asyncio
    async def test_delegation_chain_disabled(self) -> None:
        hub = Hub()
        sub_agent = _VariableCaptureAgent("synthesizer", result="done")
        coordinator = _DelegatingAgent("coordinator", hub, delegate_to="synthesizer")

        await hub.register(coordinator)
        await hub.register(sub_agent)

        await hub.ask(coordinator, "run pipeline", metadata={"project_id": "abc-123"})

        assert coordinator.received_variables is None
        assert sub_agent.received_variables is None


class TestMetadataEnabled:
    """Hub with propagate_metadata=True injects metadata as variables."""

    @pytest.mark.asyncio
    async def test_metadata_as_variables(self) -> None:
        hub = Hub(propagate_metadata=True)
        agent = _VariableCaptureAgent("worker")
        await hub.register(agent)

        await hub.ask(agent, "do task", metadata={"project_id": "abc", "env": "prod"})
        assert agent.received_variables is not None
        assert agent.received_variables["project_id"] == "abc"
        assert agent.received_variables["env"] == "prod"

    @pytest.mark.asyncio
    async def test_delegation_chain(self) -> None:
        hub = Hub(propagate_metadata=True)
        sub_agent = _VariableCaptureAgent("synthesizer", result="insights found")
        coordinator = _DelegatingAgent("coordinator", hub, delegate_to="synthesizer")

        await hub.register(coordinator)
        await hub.register(sub_agent)

        await hub.ask(coordinator, "run pipeline", metadata={"project_id": "abc-123"})

        assert coordinator.received_variables is not None
        assert coordinator.received_variables["project_id"] == "abc-123"

        assert sub_agent.received_variables is not None
        assert sub_agent.received_variables["project_id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_explicit_variables_override_metadata(self) -> None:
        hub = Hub(propagate_metadata=True)
        agent = _VariableCaptureAgent("worker")
        await hub.register(agent)

        await hub.ask(
            agent,
            "do task",
            metadata={"project_id": "from-metadata", "extra": "meta-val"},
            variables={"project_id": "from-variables"},
        )

        assert agent.received_variables["project_id"] == "from-variables"
        assert agent.received_variables["extra"] == "meta-val"

    @pytest.mark.asyncio
    async def test_network_convenience(self) -> None:
        network = Network(propagate_metadata=True)
        agent = _VariableCaptureAgent("worker")
        await network.register(agent)

        await network.ask(agent, "do task", metadata={"project_id": "xyz"})

        assert agent.received_variables is not None
        assert agent.received_variables["project_id"] == "xyz"


class TestEnvelopeMetadata:
    """Envelope.child() metadata propagation."""

    @pytest.mark.asyncio
    async def test_child_preserves_metadata(self) -> None:
        parent = Envelope(
            event=type("E", (), {"type": "test"})(),
            sender="coordinator",
            recipient="worker",
            metadata={"project_id": "abc", "env": "prod"},
        )

        child = parent.child(
            event=type("E", (), {"type": "result"})(),
            sender="worker",
            recipient="coordinator",
        )

        assert child.metadata["project_id"] == "abc"
        assert child.metadata["env"] == "prod"
        assert child.trace_id == parent.trace_id
        assert child.causation_id == parent.correlation_id

    @pytest.mark.asyncio
    async def test_child_metadata_override(self) -> None:
        parent = Envelope(
            event=type("E", (), {"type": "test"})(),
            sender="a",
            metadata={"project_id": "abc", "step": "1"},
        )

        child = parent.child(
            event=type("E", (), {"type": "test"})(),
            sender="b",
            metadata={"step": "2"},
        )

        assert child.metadata["project_id"] == "abc"  # inherited
        assert child.metadata["step"] == "2"  # overridden


_REMOTE_PORT = 19876


@_skip_no_aiohttp
class TestMetadataOverHTTP:
    """Metadata propagates across Hub boundaries via RemoteAgent HTTP delegation."""

    @pytest.mark.asyncio
    async def test_metadata_propagates_over_http(self) -> None:
        """Metadata survives the RemoteAgent → HTTP → remote Hub boundary."""
        # Hub A: hosts a worker that captures variables
        hub_a = Hub(propagate_metadata=True)
        worker = _VariableCaptureAgent("worker", result="remote work done")
        await hub_a.register(worker)

        async with hub_a.serve(host="127.0.0.1", port=_REMOTE_PORT):
            # Hub B: coordinator connects to Hub A
            hub_b = Hub(propagate_metadata=True)
            await hub_b.connect(f"http://127.0.0.1:{_REMOTE_PORT}")

            # Delegate to the remote worker with metadata
            result = await hub_b.delegate(
                "coordinator",
                "worker",
                "do remote task",
                metadata={"project_id": "abc-123"},
            )

            assert "remote work done" in result
            assert worker.received_variables is not None
            assert worker.received_variables["project_id"] == "abc-123"

            await hub_b.close()
