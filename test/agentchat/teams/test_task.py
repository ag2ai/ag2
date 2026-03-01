# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Task system - Phase 2 of Teams."""

from __future__ import annotations

import pytest

from autogen.agentchat.teams._task import Task, TaskList


class TestTask:
    """Test the Task model."""

    def test_task_creation(self) -> None:
        task = Task(id="1", subject="Test task", description="A test")
        assert task.id == "1"
        assert task.subject == "Test task"
        assert task.status == "pending"
        assert task.owner is None
        assert task.blocked_by == []
        assert task.blocks == []
        assert task.result is None
        assert task.is_available is True
        assert task.is_blocked is False

    def test_task_blocked(self) -> None:
        task = Task(id="1", subject="Blocked", blocked_by=["2"])
        assert task.is_blocked is True
        assert task.is_available is False

    def test_task_serializable(self) -> None:
        task = Task(id="1", subject="Test", description="Desc", metadata={"key": "value"})
        data = task.model_dump_json()
        restored = Task.model_validate_json(data)
        assert restored.id == "1"
        assert restored.subject == "Test"
        assert restored.metadata == {"key": "value"}


class TestTaskList:
    """Test the TaskList manager."""

    def test_create_task(self) -> None:
        tl = TaskList()
        task = tl.create("First task", "Do something")
        assert task.id == "1"
        assert task.subject == "First task"
        assert task.description == "Do something"
        assert len(tl.tasks) == 1

    def test_auto_incrementing_ids(self) -> None:
        tl = TaskList()
        t1 = tl.create("Task 1")
        t2 = tl.create("Task 2")
        t3 = tl.create("Task 3")
        assert t1.id == "1"
        assert t2.id == "2"
        assert t3.id == "3"

    def test_get_task(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        task = tl.get("1")
        assert task.subject == "Task 1"

    def test_get_task_with_hash_prefix(self) -> None:
        """LLMs often pass '#1' instead of '1' - both should work."""
        tl = TaskList()
        tl.create("Task 1")
        assert tl.get("#1").subject == "Task 1"
        tl.claim("#1", "worker")
        assert tl.get("#1").status == "in_progress"
        tl.complete("#1", "done")
        assert tl.get("#1").status == "completed"

    def test_hash_prefix_in_blocked_by(self) -> None:
        """LLMs pass '#1' in blocked_by - must normalize so unblocking works."""
        tl = TaskList()
        tl.create("Task 1")
        # LLM passes "#1" in blocked_by
        tl.create("Task 2", blocked_by=["#1"])
        assert tl.get("2").blocked_by == ["1"]  # stored normalized
        assert tl.get("2").is_blocked is True

        # Complete task 1 → task 2 should unblock
        tl.claim("1", "worker")
        unblocked = tl.complete("1", "done")
        assert len(unblocked) == 1
        assert unblocked[0].id == "2"
        assert tl.get("2").is_available is True

    def test_hash_prefix_in_complete_unblocks(self) -> None:
        """Completing with '#1' must still unblock dependents."""
        tl = TaskList()
        tl.create("Task 1")
        tl.create("Task 2", blocked_by=["1"])
        tl.claim("#1", "worker")
        unblocked = tl.complete("#1", "done")
        assert len(unblocked) == 1
        assert unblocked[0].id == "2"

    def test_get_missing_task_raises(self) -> None:
        tl = TaskList()
        with pytest.raises(KeyError, match="not found"):
            tl.get("99")

    def test_claim_task(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        task = tl.claim("1", "worker-1")
        assert task.owner == "worker-1"
        assert task.status == "in_progress"

    def test_claim_already_claimed_raises(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        with pytest.raises(ValueError, match="not pending"):
            tl.claim("1", "worker-2")

    def test_claim_blocked_task_raises(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        tl.create("Task 2", blocked_by=["1"])
        with pytest.raises(ValueError, match="blocked"):
            tl.claim("2", "worker-1")

    def test_complete_task(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        unblocked = tl.complete("1", result="Done!")
        task = tl.get("1")
        assert task.status == "completed"
        assert task.result == "Done!"
        assert unblocked == []

    def test_complete_not_in_progress_raises(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        with pytest.raises(ValueError, match="not in_progress"):
            tl.complete("1")

    def test_dependencies_basic(self) -> None:
        """Test that dependencies are set up correctly."""
        tl = TaskList()
        t1 = tl.create("Research")
        t2 = tl.create("Implement", blocked_by=["1"])
        assert t2.blocked_by == ["1"]
        assert t1.blocks == ["2"]
        assert t2.is_blocked is True
        assert t2.is_available is False

    def test_dependencies_unblock_on_complete(self) -> None:
        """Test that completing a task unblocks dependents."""
        tl = TaskList()
        tl.create("Research")
        tl.create("Implement", blocked_by=["1"])
        tl.create("Test", blocked_by=["2"])

        # Complete task 1 → should unblock task 2
        tl.claim("1", "worker-1")
        unblocked = tl.complete("1", "Research complete")

        assert len(unblocked) == 1
        assert unblocked[0].id == "2"
        assert tl.get("2").is_available is True
        assert tl.get("3").is_blocked is True  # still blocked by 2

    def test_dependencies_multiple_blockers(self) -> None:
        """Test task blocked by multiple tasks."""
        tl = TaskList()
        tl.create("Task A")
        tl.create("Task B")
        tl.create("Task C", blocked_by=["1", "2"])

        assert tl.get("3").blocked_by == ["1", "2"]
        assert tl.get("3").is_blocked is True

        # Complete A → C still blocked by B
        tl.claim("1", "worker-1")
        unblocked = tl.complete("1")
        assert len(unblocked) == 0
        assert tl.get("3").blocked_by == ["2"]
        assert tl.get("3").is_blocked is True

        # Complete B → C now unblocked
        tl.claim("2", "worker-2")
        unblocked = tl.complete("2")
        assert len(unblocked) == 1
        assert unblocked[0].id == "3"
        assert tl.get("3").is_available is True

    def test_dependencies_invalid_reference_raises(self) -> None:
        """Test that referencing a non-existent task raises."""
        tl = TaskList()
        with pytest.raises(KeyError, match="not found"):
            tl.create("Bad dep", blocked_by=["99"])

    def test_available_tasks(self) -> None:
        tl = TaskList()
        tl.create("Available 1")
        tl.create("Available 2")
        tl.create("Blocked", blocked_by=["1"])

        available = tl.available()
        assert len(available) == 2
        assert {t.id for t in available} == {"1", "2"}

        # Claim one → only 1 available
        tl.claim("1", "worker-1")
        available = tl.available()
        assert len(available) == 1
        assert available[0].id == "2"

    def test_by_status(self) -> None:
        tl = TaskList()
        tl.create("Pending 1")
        tl.create("Pending 2")
        tl.create("In Progress")
        tl.claim("3", "worker-1")

        assert len(tl.by_status("pending")) == 2
        assert len(tl.by_status("in_progress")) == 1
        assert len(tl.by_status("completed")) == 0

    def test_by_owner(self) -> None:
        tl = TaskList()
        tl.create("Task 1")
        tl.create("Task 2")
        tl.create("Task 3")
        tl.claim("1", "worker-1")
        tl.claim("2", "worker-1")
        tl.claim("3", "worker-2")

        assert len(tl.by_owner("worker-1")) == 2
        assert len(tl.by_owner("worker-2")) == 1
        assert len(tl.by_owner("worker-3")) == 0

    def test_summary(self) -> None:
        tl = TaskList()
        tl.create("Research")
        tl.create("Implement", blocked_by=["1"])
        tl.claim("1", "worker-1")

        summary = tl.summary()
        assert "#1" in summary
        assert "#2" in summary
        assert "worker-1" in summary
        assert "Research" in summary

    def test_to_dict_for_llm(self) -> None:
        tl = TaskList()
        tl.create("Task 1", "Description 1")
        tl.create("Task 2", blocked_by=["1"])

        dicts = tl.to_dict_for_llm()
        assert len(dicts) == 2
        assert dicts[0]["id"] == "1"
        assert dicts[0]["subject"] == "Task 1"
        assert dicts[1]["blocked_by"] == ["1"]

    def test_serialization_roundtrip(self) -> None:
        """Test that TaskList can be serialized and deserialized."""
        tl = TaskList()
        tl.create("Task 1", "First task")
        tl.create("Task 2", "Second task", blocked_by=["1"])
        tl.create("Task 3", metadata={"priority": "high"})
        tl.claim("1", "worker-1")
        tl.complete("1", "Done!")

        # Serialize
        data = tl.model_dump_json()

        # Deserialize
        restored = TaskList.model_validate_json(data)
        assert len(restored.tasks) == 3
        assert restored.get("1").status == "completed"
        assert restored.get("1").result == "Done!"
        assert restored.get("2").is_available is True  # unblocked when 1 completed
        assert restored.get("3").metadata == {"priority": "high"}

    def test_release_task(self) -> None:
        """Test releasing an in_progress task back to pending."""
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        assert tl.get("1").status == "in_progress"
        assert tl.get("1").owner == "worker-1"

        released = tl.release("1")
        assert released.status == "pending"
        assert released.owner is None
        assert released.is_available is True

    def test_release_not_in_progress_raises(self) -> None:
        """Test that releasing a pending task raises."""
        tl = TaskList()
        tl.create("Task 1")
        with pytest.raises(ValueError, match="not in_progress"):
            tl.release("1")

    def test_release_completed_raises(self) -> None:
        """Test that releasing a completed task raises."""
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        tl.complete("1", "Done")
        with pytest.raises(ValueError, match="not in_progress"):
            tl.release("1")

    def test_release_then_reclaim(self) -> None:
        """Test full cycle: claim → release → reclaim by different agent."""
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        tl.release("1")

        # Now a different agent can claim it
        task = tl.claim("1", "worker-2")
        assert task.owner == "worker-2"
        assert task.status == "in_progress"

    def test_release_with_hash_prefix(self) -> None:
        """Test that release normalizes '#1' to '1'."""
        tl = TaskList()
        tl.create("Task 1")
        tl.claim("1", "worker-1")
        released = tl.release("#1")
        assert released.status == "pending"
        assert released.owner is None

    def test_pipeline_pattern(self) -> None:
        """Test a full pipeline: research → plan → implement → test."""
        tl = TaskList()
        tl.create("Research", "Research best practices")
        tl.create("Plan", "Create implementation plan", blocked_by=["1"])
        tl.create("Implement", "Build the feature", blocked_by=["2"])
        tl.create("Test", "Write and run tests", blocked_by=["3"])

        # Only task 1 is available
        assert len(tl.available()) == 1
        assert tl.available()[0].id == "1"

        # Work through pipeline
        tl.claim("1", "researcher")
        unblocked = tl.complete("1", "Research findings here")
        assert len(unblocked) == 1
        assert unblocked[0].id == "2"

        tl.claim("2", "planner")
        unblocked = tl.complete("2", "Implementation plan ready")
        assert len(unblocked) == 1
        assert unblocked[0].id == "3"

        tl.claim("3", "developer")
        unblocked = tl.complete("3", "Code written")
        assert len(unblocked) == 1
        assert unblocked[0].id == "4"

        tl.claim("4", "tester")
        unblocked = tl.complete("4", "All tests pass")
        assert len(unblocked) == 0

        # All completed
        assert len(tl.by_status("completed")) == 4
        assert len(tl.available()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
