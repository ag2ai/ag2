# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-server distributed agent communication.

Tests RemoteAgent, Hub HTTP server endpoints, Hub.connect(), and
end-to-end cross-hub delegation.
"""


import pytest

from autogen.beta.network.hub import Hub
from autogen.beta.network.remote import RemoteAgent, RemoteAgentReply

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AskableAgent:
    """Mock agent that returns a canned response."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result
        self.last_message: str | None = None

    async def ask(self, message: str, **kwargs):
        self.last_message = message
        return type("Reply", (), {"content": self._result, "body": self._result})()


class _MockAgent:
    """Minimal mock agent for registry tests."""

    def __init__(self, name: str):
        self.name = name


# Use high ports to avoid conflicts with other services
_PORT_A = 18900
_PORT_B = 18901


# ---------------------------------------------------------------------------
# RemoteAgent unit tests
# ---------------------------------------------------------------------------


class TestRemoteAgent:
    def test_construction(self) -> None:
        remote = RemoteAgent("researcher", "http://localhost:8900")
        assert remote.name == "researcher"
        assert remote.endpoint == "http://localhost:8900"

    def test_construction_strips_trailing_slash(self) -> None:
        remote = RemoteAgent("researcher", "http://localhost:8900/")
        assert remote.endpoint == "http://localhost:8900"

    def test_repr(self) -> None:
        remote = RemoteAgent("researcher", "http://localhost:8900")
        assert "researcher" in repr(remote)
        assert "8900" in repr(remote)

    def test_capabilities_and_description(self) -> None:
        remote = RemoteAgent(
            "researcher",
            "http://localhost:8900",
            capabilities=["research", "analysis"],
            description="A research agent",
        )
        assert remote.capabilities == ["research", "analysis"]
        assert remote.description == "A research agent"


class TestRemoteAgentReply:
    def test_content(self) -> None:
        remote = RemoteAgent("test", "http://localhost:8900")
        reply = RemoteAgentReply("hello world", remote_agent=remote)
        assert reply.content == "hello world"

    def test_none_content(self) -> None:
        remote = RemoteAgent("test", "http://localhost:8900")
        reply = RemoteAgentReply(None, remote_agent=remote)
        assert reply.content is None


# ---------------------------------------------------------------------------
# Hub HTTP server tests
# ---------------------------------------------------------------------------


class TestHubServer:
    @pytest.mark.asyncio
    async def test_serve_starts_server(self) -> None:
        """Hub.serve(host=...) should start an HTTP server."""
        hub = Hub()
        await hub.register(_AskableAgent("worker", result="ok"))

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session, session.get(f"http://127.0.0.1:{_PORT_A}/health") as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "healthy"
                assert data["agents"] == 1

    @pytest.mark.asyncio
    async def test_serve_without_host_no_server(self) -> None:
        """Hub.serve() without host should not start an HTTP server (backward compat)."""
        hub = Hub()
        async with hub.serve() as h:
            assert h is hub
            # No server running — just lifecycle management

    @pytest.mark.asyncio
    async def test_discover_endpoint(self) -> None:
        """GET /discover should return registered local agents."""
        hub = Hub()
        await hub.register(
            _AskableAgent("researcher"),
            capabilities=["research"],
            description="A researcher",
        )
        await hub.register(
            _AskableAgent("writer"),
            capabilities=["writing"],
        )

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session, session.get(f"http://127.0.0.1:{_PORT_A}/discover") as resp:
                assert resp.status == 200
                data = await resp.json()
                agents = data["agents"]
                assert len(agents) == 2
                names = {a["name"] for a in agents}
                assert names == {"researcher", "writer"}

    @pytest.mark.asyncio
    async def test_discover_endpoint_filters_remote_agents(self) -> None:
        """GET /discover should NOT expose RemoteAgent proxies."""
        hub = Hub()
        await hub.register(_AskableAgent("local_agent"), capabilities=["local"])
        remote = RemoteAgent("remote_agent", "http://other-server:8900")
        await hub.register(remote, capabilities=["remote"])

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session, session.get(f"http://127.0.0.1:{_PORT_A}/discover") as resp:
                data = await resp.json()
                names = {a["name"] for a in data["agents"]}
                assert "local_agent" in names
                assert "remote_agent" not in names

    @pytest.mark.asyncio
    async def test_delegate_endpoint(self) -> None:
        """POST /delegate should execute the agent and return result."""
        hub = Hub()
        worker = _AskableAgent("worker", result="task completed")
        await hub.register(worker)

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session:
                payload = {"agent": "worker", "task": "do the thing"}
                async with session.post(
                    f"http://127.0.0.1:{_PORT_A}/delegate",
                    json=payload,
                ) as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert data["status"] == "ok"
                    assert data["result"] == "task completed"

            assert worker.last_message == "do the thing"

    @pytest.mark.asyncio
    async def test_delegate_endpoint_unknown_agent(self) -> None:
        """POST /delegate for unknown agent should return 404."""
        hub = Hub()

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session:
                payload = {"agent": "ghost", "task": "hello"}
                async with session.post(
                    f"http://127.0.0.1:{_PORT_A}/delegate",
                    json=payload,
                ) as resp:
                    assert resp.status == 404
                    data = await resp.json()
                    assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_delegate_endpoint_missing_fields(self) -> None:
        """POST /delegate without required fields should return 400."""
        hub = Hub()

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with (
                ClientSession() as session,
                session.post(
                    f"http://127.0.0.1:{_PORT_A}/delegate",
                    json={"agent": "worker"},  # missing 'task'
                ) as resp,
            ):
                assert resp.status == 400


# ---------------------------------------------------------------------------
# Hub.connect() tests
# ---------------------------------------------------------------------------


class TestHubConnect:
    @pytest.mark.asyncio
    async def test_connect_discovers_remote_agents(self) -> None:
        """Hub.connect() should create RemoteAgent proxies for remote agents."""
        hub_a = Hub()
        await hub_a.register(
            _AskableAgent("researcher", result="research done"),
            capabilities=["research"],
            description="A researcher",
        )
        await hub_a.register(
            _AskableAgent("analyst"),
            capabilities=["analysis"],
        )

        async with hub_a.serve(host="127.0.0.1", port=_PORT_A):
            hub_b = Hub()
            registered = await hub_b.connect(f"http://127.0.0.1:{_PORT_A}")

            assert len(registered) == 2
            assert "researcher" in registered
            assert "analyst" in registered

            # Verify they're registered as RemoteAgents
            assert isinstance(hub_b.agents["researcher"], RemoteAgent)
            assert isinstance(hub_b.agents["analyst"], RemoteAgent)

            # Verify capabilities were preserved
            discovered = await hub_b.discover("research")
            assert len(discovered) == 1
            assert discovered[0].name == "researcher"

            await hub_b.close()

    @pytest.mark.asyncio
    async def test_connect_unreachable_raises(self) -> None:
        """Hub.connect() to an unreachable endpoint should raise."""
        hub = Hub()
        with pytest.raises(Exception):
            await hub.connect("http://127.0.0.1:19999", timeout=2.0)


# ---------------------------------------------------------------------------
# End-to-end cross-hub delegation
# ---------------------------------------------------------------------------


class TestCrossHubDelegation:
    @pytest.mark.asyncio
    async def test_cross_hub_delegation(self) -> None:
        """Full end-to-end: Hub B delegates to an agent on Hub A via HTTP."""
        # Hub A hosts "researcher"
        hub_a = Hub()
        researcher = _AskableAgent("researcher", result="AI trends: transformers dominate")
        await hub_a.register(researcher, capabilities=["research"])

        async with hub_a.serve(host="127.0.0.1", port=_PORT_A):
            # Hub B connects to Hub A and gets a RemoteAgent proxy
            hub_b = Hub()
            await hub_b.connect(f"http://127.0.0.1:{_PORT_A}")

            # Delegate from Hub B to the remote researcher
            result = await hub_b.delegate("writer", "researcher", "Research AI trends")

            assert "AI trends" in result
            assert "transformers" in result
            assert researcher.last_message == "Research AI trends"

            await hub_b.close()

    @pytest.mark.asyncio
    async def test_cross_hub_with_local_and_remote_agents(self) -> None:
        """Hub with both local and remote agents can delegate to either."""
        hub_a = Hub()
        await hub_a.register(
            _AskableAgent("researcher", result="research results"),
            capabilities=["research"],
        )

        async with hub_a.serve(host="127.0.0.1", port=_PORT_A):
            hub_b = Hub()
            local_writer = _AskableAgent("writer", result="written report")
            await hub_b.register(local_writer, capabilities=["writing"])
            await hub_b.connect(f"http://127.0.0.1:{_PORT_A}")

            # Can delegate to local agent
            local_result = await hub_b.delegate("coordinator", "writer", "Write report")
            assert local_result == "written report"

            # Can delegate to remote agent
            remote_result = await hub_b.delegate("coordinator", "researcher", "Research topic")
            assert remote_result == "research results"

            # Discovery shows both
            all_agents = await hub_b.discover()
            names = {a.name for a in all_agents}
            assert names == {"writer", "researcher"}

            await hub_b.close()

    @pytest.mark.asyncio
    async def test_bidirectional_cross_hub(self) -> None:
        """Two Hubs can connect to each other for bidirectional delegation."""
        hub_a = Hub()
        await hub_a.register(
            _AskableAgent("researcher", result="research done"),
            capabilities=["research"],
        )

        hub_b = Hub()
        await hub_b.register(
            _AskableAgent("writer", result="writing done"),
            capabilities=["writing"],
        )

        async with hub_a.serve(host="127.0.0.1", port=_PORT_A), hub_b.serve(host="127.0.0.1", port=_PORT_B):
            # A connects to B
            await hub_a.connect(f"http://127.0.0.1:{_PORT_B}")
            # B connects to A
            await hub_b.connect(f"http://127.0.0.1:{_PORT_A}")

            # A can delegate to writer (on B)
            result_a = await hub_a.delegate("researcher", "writer", "Write report")
            assert result_a == "writing done"

            # B can delegate to researcher (on A)
            result_b = await hub_b.delegate("writer", "researcher", "Research topic")
            assert result_b == "research done"

    @pytest.mark.asyncio
    async def test_remote_agent_handles_failure_gracefully(self) -> None:
        """RemoteAgent returns error content when remote Hub is unreachable."""
        hub = Hub()
        remote = RemoteAgent(
            "ghost",
            "http://127.0.0.1:19999",
            timeout=2.0,
            max_retries=0,
        )
        await hub.register(remote)

        result = await hub.delegate("caller", "ghost", "hello")
        assert "error" in result.lower() or "failed" in result.lower()

        await hub.close()


# ---------------------------------------------------------------------------
# Hub.close() cleanup
# ---------------------------------------------------------------------------


class TestHubCloseRemote:
    @pytest.mark.asyncio
    async def test_close_cleans_up_remote_sessions(self) -> None:
        """Hub.close() should close RemoteAgent HTTP sessions."""
        hub = Hub()
        remote = RemoteAgent("test", "http://localhost:8900")
        await hub.register(remote)

        # Force session creation
        remote._session = None  # Ensure no session yet

        await hub.close()
        # Should not raise — close is safe even without a session

    @pytest.mark.asyncio
    async def test_serve_cleanup_stops_server(self) -> None:
        """Exiting serve() context should stop the HTTP server."""
        hub = Hub()
        await hub.register(_AskableAgent("worker"))

        async with hub.serve(host="127.0.0.1", port=_PORT_A):
            from aiohttp import ClientSession

            async with ClientSession() as session, session.get(f"http://127.0.0.1:{_PORT_A}/health") as resp:
                assert resp.status == 200

        # Server should be stopped now — connection should fail
        from aiohttp import ClientSession

        async with ClientSession() as session:
            with pytest.raises(Exception):
                async with session.get(
                    f"http://127.0.0.1:{_PORT_A}/health",
                    timeout=ClientSession._timeout,  # type: ignore[attr-defined]
                ):
                    pass
