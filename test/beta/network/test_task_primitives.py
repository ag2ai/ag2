# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task primitive round-trip coverage — Phase 4.

Every dataclass serializes through ``to_dict`` / ``from_dict`` (and the
JSON convenience helpers on :class:`TaskMetadata`) so the hub's on-disk
record can round-trip through a cold restart without loss.
"""

from __future__ import annotations

import json

import pytest

from autogen.beta.network import (
    TERMINAL_TASK_STATES,
    TaskMetadata,
    TaskPhase,
    TaskSpec,
    TaskState,
)


class TestTaskState:
    def test_values(self) -> None:
        assert TaskState.CREATED.value == "created"
        assert TaskState.RUNNING.value == "running"
        assert TaskState.PAUSED.value == "paused"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELLED.value == "cancelled"
        assert TaskState.EXPIRED.value == "expired"

    def test_terminal_states(self) -> None:
        assert TaskState.COMPLETED in TERMINAL_TASK_STATES
        assert TaskState.FAILED in TERMINAL_TASK_STATES
        assert TaskState.CANCELLED in TERMINAL_TASK_STATES
        assert TaskState.EXPIRED in TERMINAL_TASK_STATES
        assert TaskState.CREATED not in TERMINAL_TASK_STATES
        assert TaskState.RUNNING not in TERMINAL_TASK_STATES
        assert TaskState.PAUSED not in TERMINAL_TASK_STATES

    def test_is_str_subclass(self) -> None:
        # ``str(TaskState.COMPLETED)`` returns the qualified name in
        # Python 3.11+ — but equality and ``.value`` still work, which
        # is what the hub and serialization paths rely on.
        assert TaskState.COMPLETED == "completed"
        assert TaskState("completed") is TaskState.COMPLETED


class TestTaskPhase:
    def test_construct_minimal(self) -> None:
        phase = TaskPhase(id="fetch")
        assert phase.id == "fetch"
        assert phase.description == ""
        assert phase.started_at is None
        assert phase.completed_at is None

    def test_construct_with_timestamps(self) -> None:
        phase = TaskPhase(
            id="summarize",
            description="synthesize abstracts",
            started_at="2026-04-12T18:30:00Z",
            completed_at="2026-04-12T18:35:00Z",
        )
        assert phase.description == "synthesize abstracts"
        assert phase.started_at == "2026-04-12T18:30:00Z"
        assert phase.completed_at == "2026-04-12T18:35:00Z"

    def test_round_trip(self) -> None:
        phase = TaskPhase(
            id="fetch",
            description="get the papers",
            started_at="2026-04-12T18:30:00Z",
        )
        data = phase.to_dict()
        assert data == {
            "id": "fetch",
            "description": "get the papers",
            "started_at": "2026-04-12T18:30:00Z",
            "completed_at": None,
        }
        hydrated = TaskPhase.from_dict(data)
        assert hydrated == phase


class TestTaskSpec:
    def test_construct_minimal(self) -> None:
        spec = TaskSpec(title="Investigate")
        assert spec.title == "Investigate"
        assert spec.description == ""
        assert spec.phases == []
        assert spec.spec_type == ""
        assert spec.payload == {}

    def test_construct_full(self) -> None:
        phases = [TaskPhase(id="fetch"), TaskPhase(id="synthesize")]
        spec = TaskSpec(
            title="Literature review",
            description="Summarize CRISPR 2025 papers",
            phases=phases,
            spec_type="research",
            payload={"top_k": 10},
        )
        assert spec.phase_ids() == ["fetch", "synthesize"]
        assert spec.payload["top_k"] == 10

    def test_round_trip(self) -> None:
        spec = TaskSpec(
            title="Draft PR body",
            description="Write a 500-word summary",
            phases=[TaskPhase(id="outline"), TaskPhase(id="write")],
            spec_type="writing",
            payload={"word_target": 500},
        )
        data = spec.to_dict()
        hydrated = TaskSpec.from_dict(data)
        assert hydrated.title == spec.title
        assert hydrated.description == spec.description
        assert [p.id for p in hydrated.phases] == ["outline", "write"]
        assert hydrated.spec_type == "writing"
        assert hydrated.payload == {"word_target": 500}

    def test_from_dict_tolerates_missing_fields(self) -> None:
        # The hub may receive partial specs across hub upgrades — missing
        # keys should default gracefully rather than raise.
        spec = TaskSpec.from_dict({"title": "tiny"})
        assert spec.title == "tiny"
        assert spec.phases == []
        assert spec.payload == {}


class TestTaskMetadata:
    def _base(self, **overrides: object) -> TaskMetadata:
        base = dict(
            task_id="01932task...",
            session_id="01932session...",
            owner_id="01932owner...",
            requester_id="01932req...",
            spec=TaskSpec(title="t"),
            state=TaskState.CREATED,
            created_at="2026-04-12T18:30:00Z",
            expires_at="2026-04-12T18:45:00Z",
        )
        base.update(overrides)
        return TaskMetadata(**base)  # type: ignore[arg-type]

    def test_is_terminal(self) -> None:
        assert not self._base().is_terminal()
        assert not self._base(state=TaskState.RUNNING).is_terminal()
        assert self._base(state=TaskState.COMPLETED).is_terminal()
        assert self._base(state=TaskState.FAILED).is_terminal()
        assert self._base(state=TaskState.CANCELLED).is_terminal()
        assert self._base(state=TaskState.EXPIRED).is_terminal()

    def test_round_trip_minimal(self) -> None:
        task = self._base()
        hydrated = TaskMetadata.from_dict(task.to_dict())
        assert hydrated.task_id == task.task_id
        assert hydrated.session_id == task.session_id
        assert hydrated.owner_id == task.owner_id
        assert hydrated.requester_id == task.requester_id
        assert hydrated.state is TaskState.CREATED
        assert hydrated.current_phase is None
        assert hydrated.started_at is None
        assert hydrated.completed_at is None
        assert hydrated.result is None
        assert hydrated.error is None
        assert hydrated.progress == {}

    def test_round_trip_full(self) -> None:
        task = self._base(
            state=TaskState.RUNNING,
            current_phase="summarize",
            started_at="2026-04-12T18:31:00Z",
            last_progress_at="2026-04-12T18:32:00Z",
            progress={"docs": 3, "pct": 0.4},
            spec=TaskSpec(
                title="Literature review",
                description="Summarize papers",
                phases=[TaskPhase(id="fetch"), TaskPhase(id="summarize")],
                spec_type="research",
                payload={"top_k": 10},
            ),
        )
        raw = task.to_json()
        parsed = json.loads(raw)
        assert parsed["state"] == "running"
        assert parsed["current_phase"] == "summarize"
        assert parsed["spec"]["spec_type"] == "research"
        assert parsed["progress"]["docs"] == 3

        hydrated = TaskMetadata.from_json(raw)
        assert hydrated.state is TaskState.RUNNING
        assert hydrated.current_phase == "summarize"
        assert hydrated.spec.spec_type == "research"
        assert hydrated.progress == {"docs": 3, "pct": 0.4}
        assert [p.id for p in hydrated.spec.phases] == ["fetch", "summarize"]

    def test_round_trip_terminal(self) -> None:
        task = self._base(
            state=TaskState.COMPLETED,
            started_at="2026-04-12T18:31:00Z",
            completed_at="2026-04-12T18:44:00Z",
            result={"summary": "done"},
        )
        hydrated = TaskMetadata.from_dict(task.to_dict())
        assert hydrated.is_terminal()
        assert hydrated.state is TaskState.COMPLETED
        assert hydrated.result == {"summary": "done"}

    def test_round_trip_failed(self) -> None:
        task = self._base(
            state=TaskState.FAILED,
            started_at="2026-04-12T18:31:00Z",
            completed_at="2026-04-12T18:33:00Z",
            error="RuntimeError: model timed out",
        )
        hydrated = TaskMetadata.from_dict(task.to_dict())
        assert hydrated.state is TaskState.FAILED
        assert hydrated.error == "RuntimeError: model timed out"

    def test_copy_is_independent(self) -> None:
        task = self._base(progress={"a": 1})
        clone = task.copy()
        clone.progress["b"] = 2
        assert task.progress == {"a": 1}
        assert clone.progress == {"a": 1, "b": 2}

    def test_from_dict_rejects_missing_required_keys(self) -> None:
        with pytest.raises(KeyError):
            TaskMetadata.from_dict({"task_id": "x"})
