# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.network.hub import Hub
from autogen.beta.network.plugins.telemetry import DelegationMetrics, TelemetryPlugin


class _MockAgent:
    """Minimal mock agent for telemetry tests (no LLM calls)."""

    def __init__(self, name: str, result: str = "done"):
        self.name = name
        self._result = result

    async def ask(self, message, **kwargs):
        return type("Reply", (), {"content": self._result, "body": self._result})()


class TestTelemetryPluginInstall:
    @pytest.mark.asyncio
    async def test_install_subscribes_to_hub_stream(self) -> None:
        """install() should add subscriptions to the hub's stream."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])

        # After install, the plugin should have 2 subscription IDs
        assert len(telemetry._sub_ids) == 2
        assert telemetry._hub is hub

    @pytest.mark.asyncio
    async def test_install_creates_fresh_metrics(self) -> None:
        telemetry = TelemetryPlugin()
        Hub(plugins=[telemetry])

        assert isinstance(telemetry.metrics, DelegationMetrics)
        assert telemetry.metrics.total_delegations == 0
        assert telemetry.metrics.total_completions == 0


class TestTelemetryPluginUninstall:
    @pytest.mark.asyncio
    async def test_uninstall_cleans_up_subscriptions(self) -> None:
        """uninstall() should clear subscription IDs and hub reference."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])

        assert len(telemetry._sub_ids) == 2
        assert telemetry._hub is hub

        telemetry.uninstall()

        assert len(telemetry._sub_ids) == 0
        assert telemetry._hub is None

    @pytest.mark.asyncio
    async def test_uninstall_removes_stream_subscriptions(self) -> None:
        """After uninstall, the hub stream should no longer have the plugin's subscriptions."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])

        sub_ids = list(telemetry._sub_ids)
        telemetry.uninstall()

        # Verify subscription IDs are no longer in the stream's subscriber dict
        for sub_id in sub_ids:
            assert sub_id not in hub.stream._subscribers


class TestTelemetryMetricsDelegationRequest:
    @pytest.mark.asyncio
    async def test_tracks_total_delegations(self) -> None:
        """DelegationRequest events should increment total_delegations."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        await hub.delegate("caller", "worker", "task-1")
        assert telemetry.metrics.total_delegations == 1

        await hub.delegate("caller", "worker", "task-2")
        assert telemetry.metrics.total_delegations == 2

    @pytest.mark.asyncio
    async def test_tracks_total_completions(self) -> None:
        """DelegationResult events should increment total_completions."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        await hub.delegate("caller", "worker", "task-1")
        assert telemetry.metrics.total_completions == 1

        await hub.delegate("caller", "worker", "task-2")
        assert telemetry.metrics.total_completions == 2

    @pytest.mark.asyncio
    async def test_tracks_by_source(self) -> None:
        """by_source should count delegations per source agent."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        await hub.delegate("alice", "worker", "task-1")
        await hub.delegate("alice", "worker", "task-2")
        await hub.delegate("bob", "worker", "task-3")

        assert telemetry.metrics.by_source["alice"] == 2
        assert telemetry.metrics.by_source["bob"] == 1

    @pytest.mark.asyncio
    async def test_tracks_by_target(self) -> None:
        """by_target should count delegations per target agent."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker-a"))
        await hub.register(_MockAgent("worker-b"))

        await hub.delegate("caller", "worker-a", "task-1")
        await hub.delegate("caller", "worker-a", "task-2")
        await hub.delegate("caller", "worker-b", "task-3")

        assert telemetry.metrics.by_target["worker-a"] == 2
        assert telemetry.metrics.by_target["worker-b"] == 1

    @pytest.mark.asyncio
    async def test_tracks_last_delegation_time(self) -> None:
        """last_delegation_time should be set after a delegation."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        assert telemetry.metrics.last_delegation_time is None

        await hub.delegate("caller", "worker", "task-1")

        assert telemetry.metrics.last_delegation_time is not None
        assert isinstance(telemetry.metrics.last_delegation_time, float)

    @pytest.mark.asyncio
    async def test_last_delegation_time_updates(self) -> None:
        """last_delegation_time should update on each new delegation."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        await hub.delegate("caller", "worker", "task-1")
        first_time = telemetry.metrics.last_delegation_time

        await hub.delegate("caller", "worker", "task-2")
        second_time = telemetry.metrics.last_delegation_time

        assert second_time is not None
        assert first_time is not None
        assert second_time >= first_time


class TestTelemetryPluginInstallUninstallCycle:
    @pytest.mark.asyncio
    async def test_no_tracking_after_uninstall(self) -> None:
        """After install then uninstall, no more metrics should be tracked."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("worker"))

        # First delegation is tracked
        await hub.delegate("caller", "worker", "task-1")
        assert telemetry.metrics.total_delegations == 1
        assert telemetry.metrics.total_completions == 1

        # Uninstall
        telemetry.uninstall()

        # Second delegation should NOT update metrics
        await hub.delegate("caller", "worker", "task-2")
        assert telemetry.metrics.total_delegations == 1
        assert telemetry.metrics.total_completions == 1
        assert telemetry.metrics.by_source["caller"] == 1
        assert telemetry.metrics.by_target["worker"] == 1

    @pytest.mark.asyncio
    async def test_hub_close_uninstalls_plugin(self) -> None:
        """Hub.close() should call uninstall on the telemetry plugin."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])

        assert telemetry._hub is hub

        await hub.close()

        assert telemetry._hub is None
        assert len(telemetry._sub_ids) == 0


class TestTelemetryPluginMultipleSources:
    @pytest.mark.asyncio
    async def test_combined_metrics_across_sources_and_targets(self) -> None:
        """Verify metrics aggregate correctly across multiple sources and targets."""
        telemetry = TelemetryPlugin()
        hub = Hub(plugins=[telemetry])
        await hub.register(_MockAgent("research"))
        await hub.register(_MockAgent("writer"))

        await hub.delegate("alice", "research", "find info")
        await hub.delegate("bob", "writer", "draft report")
        await hub.delegate("alice", "writer", "edit report")

        assert telemetry.metrics.total_delegations == 3
        assert telemetry.metrics.total_completions == 3
        assert telemetry.metrics.by_source["alice"] == 2
        assert telemetry.metrics.by_source["bob"] == 1
        assert telemetry.metrics.by_target["research"] == 1
        assert telemetry.metrics.by_target["writer"] == 2
