# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hub's background (non-blocking, StateStore-backed) delegation API.

Covers:
- request_background spawns, returns an id, persists initial state
- check_delegation / list_pending / cancel_delegation lifecycle
- DelegationStarted / Progress / Cancelled events emitted on the Hub stream
- recover_orphaned_delegations flags pending/running state from a prior process
- StateStore.scan used to enumerate delegations
- Hub.close cancels in-flight background tasks
- Network tool actions (request_background, check_status, cancel, list_pending,
  report_progress) route through the Hub correctly
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from autogen.beta.network.events import (
    DelegationCancelled,
    DelegationError,
    DelegationProgress,
    DelegationRequest,
    DelegationResult,
    DelegationStarted,
)
from autogen.beta.network.hub import (
    _DELEGATION_KEY_PREFIX,
    Hub,
    _current_delegation_id,
    _delegation_key,
    _format_delegation_state,
)
from autogen.beta.state import MemoryStateStore


class _Reply:
    def __init__(self, body: str) -> None:
        self.body = body
        self.content = body


class _SlowAgent:
    """Agent whose ask() sleeps before returning, useful for cancel/check_status."""

    def __init__(self, name: str, delay: float = 0.05, result: str = "done") -> None:
        self.name = name
        self._delay = delay
        self._result = result
        self.ask_started: asyncio.Event = asyncio.Event()
        self.ask_cancelled: bool = False

    async def ask(self, message: str, **kwargs: Any) -> _Reply:
        self.ask_started.set()
        try:
            await asyncio.sleep(self._delay)
        except asyncio.CancelledError:
            self.ask_cancelled = True
            raise
        return _Reply(self._result)


class _FastAgent:
    def __init__(self, name: str, result: str = "ok") -> None:
        self.name = name
        self._result = result

    async def ask(self, message: str, **kwargs: Any) -> _Reply:
        return _Reply(self._result)


class _ExplodingAgent:
    def __init__(self, name: str, error: str = "kaboom") -> None:
        self.name = name
        self._error = error

    async def ask(self, message: str, **kwargs: Any) -> _Reply:
        raise RuntimeError(self._error)


class _ProgressReportingAgent:
    """Calls hub.report_progress via contextvar before finishing."""

    def __init__(self, name: str, hub: Hub, result: str = "progress done") -> None:
        self.name = name
        self._hub = hub
        self._result = result

    async def ask(self, message: str, **kwargs: Any) -> _Reply:
        did = _current_delegation_id.get()
        await self._hub.report_progress(did, progress=0.5, message="halfway")
        return _Reply(self._result)


async def _collect_events(hub: Hub, event_types: tuple[type, ...]) -> list[Any]:
    """Subscribe to the Hub stream and return a list that captures matching events."""
    captured: list[Any] = []

    async def _sub(event: Any) -> None:
        if isinstance(event, event_types):
            captured.append(event)

    hub.stream.subscribe(_sub)
    return captured


# ---------------------------------------------------------------------------
# StateStore scan protocol (on the vendor-neutral MemoryStateStore)
# ---------------------------------------------------------------------------


class TestStateStoreScan:
    @pytest.mark.asyncio
    async def test_scan_returns_prefix_matching_keys(self) -> None:
        store = MemoryStateStore()
        await store.set("delegation:a", {"id": "a"})
        await store.set("delegation:b", {"id": "b"})
        await store.set("other:c", {"id": "c"})

        keys = await store.scan("delegation:")
        assert set(keys) == {"delegation:a", "delegation:b"}

    @pytest.mark.asyncio
    async def test_scan_excludes_expired(self) -> None:
        store = MemoryStateStore()
        # Insert an already-expired TTL entry to force it through the expiry
        # cleanup path inside scan.
        await store.set("delegation:a", {"id": "a"}, ttl=0.0)
        await asyncio.sleep(0.01)
        await store.set("delegation:b", {"id": "b"})

        keys = await store.scan("delegation:")
        assert keys == ["delegation:b"]
        # And the expired key is gone after scan
        assert await store.get("delegation:a") is None

    @pytest.mark.asyncio
    async def test_scan_empty(self) -> None:
        store = MemoryStateStore()
        assert await store.scan("nothing:") == []


# ---------------------------------------------------------------------------
# request_background — happy path, persistence, events
# ---------------------------------------------------------------------------


class TestRequestBackground:
    @pytest.mark.asyncio
    async def test_returns_id_and_persists_initial_state(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker", result="hello"))

        delegation_id = await hub.request_background(
            "worker", "do the thing", source="pilot"
        )
        assert delegation_id
        # Initial state should be readable even before the task finishes;
        # give it a yield so it can advance past ``pending``.
        state = await hub.check_delegation(delegation_id)
        assert state is not None
        assert state["delegation_id"] == delegation_id
        assert state["target"] == "worker"
        assert state["source"] == "pilot"
        assert state["task"] == "do the thing"
        assert state["state"] in {"pending", "running", "completed"}

        # Now await completion and confirm terminal state
        await asyncio.sleep(0.05)
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "completed"
        assert state["result"] == "hello"
        assert state["error"] is None
        assert state["created_at"] <= state["updated_at"]
        await hub.close()

    @pytest.mark.asyncio
    async def test_emits_delegation_started(self) -> None:
        hub = Hub()
        captured = await _collect_events(hub, (DelegationStarted,))
        await hub.register(_FastAgent("worker"))
        delegation_id = await hub.request_background("worker", "task", source="pilot")
        await asyncio.sleep(0.05)
        assert any(
            e.delegation_id == delegation_id
            and e.source == "pilot"
            and e.target == "worker"
            for e in captured
        )
        await hub.close()

    @pytest.mark.asyncio
    async def test_stamps_delegation_id_on_request_and_result_events(self) -> None:
        hub = Hub()
        captured_req: list[DelegationRequest] = []
        captured_res: list[DelegationResult] = []

        async def _sub(event: Any) -> None:
            if isinstance(event, DelegationRequest):
                captured_req.append(event)
            elif isinstance(event, DelegationResult):
                captured_res.append(event)

        hub.stream.subscribe(_sub)

        await hub.register(_FastAgent("worker", result="done"))
        delegation_id = await hub.request_background("worker", "t", source="pilot")
        await asyncio.sleep(0.05)

        assert any(e.delegation_id == delegation_id for e in captured_req)
        assert any(e.delegation_id == delegation_id for e in captured_res)
        await hub.close()

    @pytest.mark.asyncio
    async def test_missing_target_raises(self) -> None:
        hub = Hub()
        with pytest.raises(ValueError):
            await hub.request_background("", "task")
        await hub.close()

    @pytest.mark.asyncio
    async def test_missing_task_raises(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker"))
        with pytest.raises(ValueError):
            await hub.request_background("worker", "")
        await hub.close()

    @pytest.mark.asyncio
    async def test_failed_agent_marks_state_failed(self) -> None:
        hub = Hub()
        await hub.register(_ExplodingAgent("bomb"))
        delegation_id = await hub.request_background("bomb", "t", source="pilot")
        await asyncio.sleep(0.05)
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "failed"
        assert state["error"] and "kaboom" in state["error"]
        await hub.close()

    @pytest.mark.asyncio
    async def test_unknown_target_marks_state_failed(self) -> None:
        hub = Hub()
        delegation_id = await hub.request_background("ghost", "t", source="pilot")
        await asyncio.sleep(0.05)
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "failed"
        assert "not found" in state["error"]
        await hub.close()

    @pytest.mark.asyncio
    async def test_uses_shared_state_store(self) -> None:
        """The Hub must persist through the StateStore passed in, not a new one."""
        store = MemoryStateStore()
        hub = Hub(state_store=store)
        await hub.register(_FastAgent("worker", result="x"))
        delegation_id = await hub.request_background("worker", "t", source="src")
        await asyncio.sleep(0.05)

        # Key should exist on the externally owned store
        raw = await store.get(_delegation_key(delegation_id))
        assert raw is not None
        assert raw["delegation_id"] == delegation_id
        await hub.close()


# ---------------------------------------------------------------------------
# check_delegation / list_pending_delegations
# ---------------------------------------------------------------------------


class TestQueryDelegation:
    @pytest.mark.asyncio
    async def test_check_unknown_returns_none(self) -> None:
        hub = Hub()
        assert await hub.check_delegation("does-not-exist") is None
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_filters_terminal_by_default(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("a", result="first"))
        await hub.register(_FastAgent("b", result="second"))
        id_a = await hub.request_background("a", "ta", source="pilot")
        id_b = await hub.request_background("b", "tb", source="pilot")
        await asyncio.sleep(0.05)

        entries = await hub.list_pending_delegations(source="pilot")
        # Both have completed — default should omit them
        assert entries == []

        with_terminal = await hub.list_pending_delegations(
            source="pilot", include_terminal=True
        )
        ids = {e["delegation_id"] for e in with_terminal}
        assert id_a in ids and id_b in ids
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_source_filter(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("a"))
        await hub.register(_FastAgent("b"))
        id_a = await hub.request_background("a", "t", source="pilot")
        id_b = await hub.request_background("b", "t", source="other")
        await asyncio.sleep(0.05)

        entries = await hub.list_pending_delegations(
            source="pilot", include_terminal=True
        )
        ids = {e["delegation_id"] for e in entries}
        assert id_a in ids and id_b not in ids
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_stable_order(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("a"))
        ids = []
        for _ in range(3):
            ids.append(await hub.request_background("a", "t", source="pilot"))
            await asyncio.sleep(0.001)
        await asyncio.sleep(0.05)
        entries = await hub.list_pending_delegations(
            source="pilot", include_terminal=True
        )
        assert [e["delegation_id"] for e in entries] == ids
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_store_without_scan_returns_empty(self) -> None:
        """A store that omits scan() shouldn't break list_pending."""

        class _NoScanStore:
            def __init__(self):
                self._d: dict[str, Any] = {}

            async def get(self, key: str):
                return self._d.get(key)

            async def set(self, key, value, ttl=None):
                self._d[key] = value

            async def delete(self, key):
                self._d.pop(key, None)

            async def exists(self, key):
                return key in self._d

        hub = Hub(state_store=_NoScanStore())
        await hub.register(_FastAgent("a"))
        await hub.request_background("a", "t", source="pilot")
        await asyncio.sleep(0.05)
        entries = await hub.list_pending_delegations(
            source="pilot", include_terminal=True
        )
        assert entries == []
        await hub.close()


# ---------------------------------------------------------------------------
# cancel_delegation
# ---------------------------------------------------------------------------


class TestCancelDelegation:
    @pytest.mark.asyncio
    async def test_cancel_in_flight(self) -> None:
        hub = Hub()
        slow = _SlowAgent("slow", delay=10.0)
        await hub.register(slow)
        delegation_id = await hub.request_background("slow", "t", source="pilot")
        await slow.ask_started.wait()

        ok = await hub.cancel_delegation(delegation_id)
        assert ok is True
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "cancelled"
        await hub.close()

    @pytest.mark.asyncio
    async def test_cancel_unknown_returns_false(self) -> None:
        hub = Hub()
        assert await hub.cancel_delegation("ghost") is False
        await hub.close()

    @pytest.mark.asyncio
    async def test_cancel_already_finished_returns_false(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("a"))
        delegation_id = await hub.request_background("a", "t", source="pilot")
        await asyncio.sleep(0.05)
        ok = await hub.cancel_delegation(delegation_id)
        # Already completed — not cancelled
        assert ok is False
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "completed"
        await hub.close()

    @pytest.mark.asyncio
    async def test_cancel_emits_cancelled_event(self) -> None:
        hub = Hub()
        captured = await _collect_events(hub, (DelegationCancelled,))
        slow = _SlowAgent("slow", delay=10.0)
        await hub.register(slow)
        delegation_id = await hub.request_background("slow", "t", source="pilot")
        await slow.ask_started.wait()
        await hub.cancel_delegation(delegation_id)
        assert any(e.delegation_id == delegation_id for e in captured)
        await hub.close()


# ---------------------------------------------------------------------------
# Hub.close cancels background tasks
# ---------------------------------------------------------------------------


class TestCloseCleanup:
    @pytest.mark.asyncio
    async def test_close_cancels_running_background_tasks(self) -> None:
        hub = Hub()
        slow = _SlowAgent("slow", delay=10.0)
        await hub.register(slow)
        delegation_id = await hub.request_background("slow", "t", source="pilot")
        await slow.ask_started.wait()

        await hub.close()

        # Agent should have had its ask cancelled
        assert slow.ask_cancelled is True
        # State should have been updated before the task unwound
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "cancelled"


# ---------------------------------------------------------------------------
# recover_orphaned_delegations
# ---------------------------------------------------------------------------


class TestRecoverOrphans:
    @pytest.mark.asyncio
    async def test_marks_pending_and_running_as_orphaned(self) -> None:
        store = MemoryStateStore()
        # Seed store as if a previous process left entries behind
        await store.set(
            _delegation_key("a"),
            {
                "delegation_id": "a",
                "state": "pending",
                "source": "pilot",
                "target": "worker",
                "task": "tA",
                "metadata": {},
                "created_at": time.time() - 10,
                "updated_at": time.time() - 5,
                "result": None,
                "error": None,
                "progress": None,
                "progress_message": "",
            },
        )
        await store.set(
            _delegation_key("b"),
            {
                "delegation_id": "b",
                "state": "running",
                "source": "pilot",
                "target": "worker",
                "task": "tB",
                "metadata": {},
                "created_at": time.time() - 10,
                "updated_at": time.time() - 5,
                "result": None,
                "error": None,
                "progress": None,
                "progress_message": "",
            },
        )
        await store.set(
            _delegation_key("c"),
            {
                "delegation_id": "c",
                "state": "completed",
                "source": "pilot",
                "target": "worker",
                "task": "tC",
                "metadata": {},
                "created_at": time.time() - 10,
                "updated_at": time.time() - 5,
                "result": "done",
                "error": None,
                "progress": None,
                "progress_message": "",
            },
        )

        hub = Hub(state_store=store)
        captured = await _collect_events(hub, (DelegationError,))
        recovered = await hub.recover_orphaned_delegations()

        assert set(recovered) == {"a", "b"}
        state_a = await store.get(_delegation_key("a"))
        state_b = await store.get(_delegation_key("b"))
        state_c = await store.get(_delegation_key("c"))
        assert state_a["state"] == "orphaned"
        assert state_b["state"] == "orphaned"
        assert state_c["state"] == "completed"  # untouched
        assert any(e.delegation_id in {"a", "b"} for e in captured)
        await hub.close()

    @pytest.mark.asyncio
    async def test_recover_with_no_scan_capability(self) -> None:
        class _NoScanStore:
            def __init__(self):
                self._d: dict[str, Any] = {}

            async def get(self, key: str):
                return self._d.get(key)

            async def set(self, key, value, ttl=None):
                self._d[key] = value

            async def delete(self, key):
                self._d.pop(key, None)

            async def exists(self, key):
                return key in self._d

        hub = Hub(state_store=_NoScanStore())
        recovered = await hub.recover_orphaned_delegations()
        assert recovered == []
        await hub.close()

    @pytest.mark.asyncio
    async def test_recover_empty_store(self) -> None:
        hub = Hub()
        recovered = await hub.recover_orphaned_delegations()
        assert recovered == []
        await hub.close()


# ---------------------------------------------------------------------------
# report_progress
# ---------------------------------------------------------------------------


class TestReportProgress:
    @pytest.mark.asyncio
    async def test_progress_updates_state_and_emits_event(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker"))
        delegation_id = await hub.request_background("worker", "t", source="pilot")
        await asyncio.sleep(0.01)

        captured = await _collect_events(hub, (DelegationProgress,))
        ok = await hub.report_progress(
            delegation_id, progress=0.25, message="quarter"
        )
        assert ok is True
        state = await hub.check_delegation(delegation_id)
        assert state["progress"] == 0.25
        assert state["progress_message"] == "quarter"
        assert any(
            e.delegation_id == delegation_id and e.progress == 0.25 for e in captured
        )
        await hub.close()

    @pytest.mark.asyncio
    async def test_progress_unknown_returns_false(self) -> None:
        hub = Hub()
        ok = await hub.report_progress("ghost", progress=0.5, message="huh")
        assert ok is False
        await hub.close()

    @pytest.mark.asyncio
    async def test_progress_via_contextvar(self) -> None:
        """Agents inside _run_background_delegation can call report_progress without an id."""
        hub = Hub()
        agent = _ProgressReportingAgent("agent", hub, result="fin")
        await hub.register(agent)
        delegation_id = await hub.request_background("agent", "t", source="pilot")
        await asyncio.sleep(0.05)

        state = await hub.check_delegation(delegation_id)
        assert state["progress"] == 0.5
        assert state["progress_message"] == "halfway"
        assert state["state"] == "completed"
        await hub.close()


# ---------------------------------------------------------------------------
# Network tool actions
# ---------------------------------------------------------------------------


class TestNetworkToolActions:
    def _network_tool(self, hub: Hub, caller: str):
        tools = hub._build_network_tools(caller=caller)
        assert len(tools) == 1
        # FunctionTool wraps the raw async closure in a CallModel (via the
        # ``@tool`` decorator). ``.model.call`` is the underlying function,
        # which lets tests skip the dependency-injection pipeline.
        return tools[0].model.call

    @pytest.mark.asyncio
    async def test_request_background_via_tool(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker", result="yes"))
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="request_background", target="worker", message="task")
        assert "Spawned background delegation" in out
        assert "delegation_id" in out
        await hub.close()

    @pytest.mark.asyncio
    async def test_request_background_rejects_self_delegation(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="request_background", target="pilot", message="task")
        assert out.startswith("Error")
        await hub.close()

    @pytest.mark.asyncio
    async def test_request_background_requires_target_and_message(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        assert (await fn(action="request_background", target="", message="t")).startswith("Error")
        assert (await fn(action="request_background", target="worker", message="")).startswith("Error")
        await hub.close()

    @pytest.mark.asyncio
    async def test_check_status_and_cancel_via_tool(self) -> None:
        hub = Hub()
        slow = _SlowAgent("slow", delay=10.0)
        await hub.register(slow)
        fn = self._network_tool(hub, "pilot")
        spawn = await fn(action="request_background", target="slow", message="t")
        delegation_id = spawn.split("delegation_id:")[-1].strip()
        await slow.ask_started.wait()

        status = await fn(action="check_status", message=delegation_id)
        assert "state:" in status
        assert delegation_id in status

        cancel = await fn(action="cancel", message=delegation_id)
        assert "Cancelled" in cancel

        final = await fn(action="check_status", message=delegation_id)
        assert "cancelled" in final
        await hub.close()

    @pytest.mark.asyncio
    async def test_check_status_unknown(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="check_status", message="ghost")
        assert "not found" in out
        await hub.close()

    @pytest.mark.asyncio
    async def test_cancel_missing_message(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="cancel", message="")
        assert out.startswith("Error")
        await hub.close()

    @pytest.mark.asyncio
    async def test_cancel_already_finished_reports_final_state(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker", result="done"))
        fn = self._network_tool(hub, "pilot")
        spawn = await fn(action="request_background", target="worker", message="t")
        delegation_id = spawn.split("delegation_id:")[-1].strip()
        await asyncio.sleep(0.05)
        out = await fn(action="cancel", message=delegation_id)
        assert "already finished" in out
        assert "completed" in out
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_via_tool(self) -> None:
        hub = Hub()
        slow = _SlowAgent("slow", delay=10.0)
        await hub.register(slow)
        fn = self._network_tool(hub, "pilot")
        spawn = await fn(action="request_background", target="slow", message="work")
        delegation_id = spawn.split("delegation_id:")[-1].strip()
        await slow.ask_started.wait()

        out = await fn(action="list_pending")
        assert delegation_id in out
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_empty(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="list_pending")
        assert "No pending delegations" in out
        await hub.close()

    @pytest.mark.asyncio
    async def test_list_pending_all_includes_terminal(self) -> None:
        hub = Hub()
        await hub.register(_FastAgent("worker", result="done"))
        fn = self._network_tool(hub, "pilot")
        spawn = await fn(action="request_background", target="worker", message="t")
        delegation_id = spawn.split("delegation_id:")[-1].strip()
        await asyncio.sleep(0.05)

        out_default = await fn(action="list_pending")
        assert delegation_id not in out_default

        out_all = await fn(action="list_pending", target="all")
        assert delegation_id in out_all
        await hub.close()

    @pytest.mark.asyncio
    async def test_report_progress_outside_delegation_errors(self) -> None:
        hub = Hub()
        fn = self._network_tool(hub, "pilot")
        out = await fn(action="report_progress", message="hi")
        assert out.startswith("Error")
        await hub.close()

    @pytest.mark.asyncio
    async def test_report_progress_via_tool_inside_delegation(self) -> None:
        """Agent calls report_progress through the network tool."""
        hub = Hub()

        class _ToolProgressAgent:
            def __init__(self, name: str):
                self.name = name

            async def ask(self, message: str, **kwargs: Any) -> _Reply:
                tools = kwargs.get("tools") or []
                assert tools, "network tools should be injected during delegation"
                # Network tool is the only one injected
                network_fn = tools[0].model.call
                await network_fn(
                    action="report_progress", topic="0.75", message="almost"
                )
                return _Reply("complete")

        await hub.register(_ToolProgressAgent("agent"))
        delegation_id = await hub.request_background("agent", "t", source="pilot")
        await asyncio.sleep(0.05)

        state = await hub.check_delegation(delegation_id)
        assert state["progress"] == 0.75
        assert state["progress_message"] == "almost"
        assert state["state"] == "completed"
        await hub.close()

    @pytest.mark.asyncio
    async def test_report_progress_invalid_topic_value(self) -> None:
        """report_progress coerces topic to float; invalid values error."""
        hub = Hub()

        class _BadProgressAgent:
            def __init__(self, name: str):
                self.name = name
                self.last_output: str = ""

            async def ask(self, message: str, **kwargs: Any) -> _Reply:
                tools = kwargs.get("tools") or []
                network_fn = tools[0].model.call
                self.last_output = await network_fn(
                    action="report_progress", topic="not-a-number", message="oops"
                )
                return _Reply("done")

        agent = _BadProgressAgent("agent")
        await hub.register(agent)
        delegation_id = await hub.request_background("agent", "t", source="pilot")
        await asyncio.sleep(0.05)
        assert agent.last_output.startswith("Error")
        state = await hub.check_delegation(delegation_id)
        assert state["state"] == "completed"
        await hub.close()


# ---------------------------------------------------------------------------
# _format_delegation_state helper
# ---------------------------------------------------------------------------


class TestFormatDelegationState:
    def test_formats_minimal_state(self) -> None:
        out = _format_delegation_state(
            {
                "delegation_id": "abc",
                "state": "pending",
                "target": "worker",
                "source": "pilot",
                "task": "do it",
            }
        )
        assert "delegation_id: abc" in out
        assert "state:         pending" in out
        assert "task:          do it" in out

    def test_truncates_long_task(self) -> None:
        long_task = "x" * 400
        out = _format_delegation_state(
            {
                "delegation_id": "abc",
                "state": "running",
                "target": "worker",
                "source": "pilot",
                "task": long_task,
            }
        )
        assert "..." in out
        assert len([l for l in out.splitlines() if l.startswith("task:")]) == 1

    def test_includes_progress_when_set(self) -> None:
        out = _format_delegation_state(
            {
                "delegation_id": "abc",
                "state": "running",
                "target": "worker",
                "source": "pilot",
                "task": "t",
                "progress": 0.3,
                "progress_message": "step 3/10",
            }
        )
        assert "30%" in out
        assert "step 3/10" in out

    def test_includes_result_and_error(self) -> None:
        out_done = _format_delegation_state(
            {
                "delegation_id": "abc",
                "state": "completed",
                "target": "worker",
                "source": "pilot",
                "task": "t",
                "result": "hi",
            }
        )
        assert "result:        hi" in out_done

        out_err = _format_delegation_state(
            {
                "delegation_id": "abc",
                "state": "failed",
                "target": "worker",
                "source": "pilot",
                "task": "t",
                "error": "boom",
            }
        )
        assert "error:         boom" in out_err


# ---------------------------------------------------------------------------
# Delegation ID present on regular (blocking) DelegationRequest/Result
# ---------------------------------------------------------------------------


class TestDelegationIdPropagation:
    @pytest.mark.asyncio
    async def test_blocking_delegation_has_empty_delegation_id(self) -> None:
        hub = Hub()
        captured_reqs: list[DelegationRequest] = []

        async def _sub(event: Any) -> None:
            if isinstance(event, DelegationRequest):
                captured_reqs.append(event)

        hub.stream.subscribe(_sub)
        await hub.register(_FastAgent("worker", result="x"))
        await hub._delegate("worker", "t", source="pilot")
        assert captured_reqs
        assert captured_reqs[-1].delegation_id == ""
        await hub.close()

    @pytest.mark.asyncio
    async def test_background_delegation_has_matching_id(self) -> None:
        hub = Hub()
        captured_reqs: list[DelegationRequest] = []

        async def _sub(event: Any) -> None:
            if isinstance(event, DelegationRequest):
                captured_reqs.append(event)

        hub.stream.subscribe(_sub)
        await hub.register(_FastAgent("worker", result="x"))
        delegation_id = await hub.request_background("worker", "t", source="pilot")
        await asyncio.sleep(0.05)
        matching = [e for e in captured_reqs if e.delegation_id == delegation_id]
        assert matching
        await hub.close()
