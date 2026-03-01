# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Task system for team orchestration.

Tasks are serializable work items with statuses, ownership, dependencies, and results.
A TaskList manages a collection of tasks with auto-incrementing IDs and dependency tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class Task(BaseModel):
    """A single unit of work in the team system.

    Tasks have a lifecycle: pending → in_progress → completed.
    They can be blocked by other tasks and owned by agents.
    """

    id: str
    subject: str
    description: str = ""
    status: Literal["pending", "in_progress", "completed"] = "pending"
    owner: str | None = None
    blocked_by: list[str] = Field(default_factory=list)
    blocks: list[str] = Field(default_factory=list)
    result: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_blocked(self) -> bool:
        """Check if this task is blocked by any incomplete tasks."""
        return len(self.blocked_by) > 0

    @property
    def is_available(self) -> bool:
        """Check if this task can be claimed (pending, unblocked, unowned)."""
        return self.status == "pending" and not self.is_blocked and self.owner is None

    def _touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)


class TaskList(BaseModel):
    """A managed collection of tasks with auto-incrementing IDs.

    Provides operations for creating, claiming, completing tasks,
    and managing dependencies between them.
    """

    tasks: list[Task] = Field(default_factory=list)
    _next_id: int = 1

    def model_post_init(self, __context: Any) -> None:
        """Set next_id based on existing tasks."""
        if self.tasks:
            max_id = max(int(t.id) for t in self.tasks)
            self._next_id = max_id + 1

    @staticmethod
    def _normalize_id(task_id: str) -> str:
        """Normalize a task ID by stripping '#' prefix and whitespace.

        LLMs often pass '#1' instead of '1' because our output formats
        IDs as '#1'. This ensures both formats work everywhere.
        """
        return task_id.strip().lstrip("#")

    def _get_next_id(self) -> str:
        """Get the next available task ID."""
        task_id = str(self._next_id)
        self._next_id += 1
        return task_id

    def get(self, task_id: str) -> Task:
        """Get a task by ID.

        Accepts IDs with or without '#' prefix (e.g. "1" or "#1").

        Raises:
            KeyError: If the task doesn't exist.
        """
        task_id = self._normalize_id(task_id)
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Task '{task_id}' not found")

    def create(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task.

        Args:
            subject: Brief title for the task.
            description: Detailed description of what needs to be done.
            blocked_by: List of task IDs that must complete before this one.
            metadata: Arbitrary key-value data to attach.

        Returns:
            The newly created Task.

        Raises:
            KeyError: If any blocked_by task ID doesn't exist.
        """
        # Normalize blocked_by IDs (LLMs pass "#1" instead of "1")
        if blocked_by:
            blocked_by = [self._normalize_id(dep_id) for dep_id in blocked_by]
            for dep_id in blocked_by:
                self.get(dep_id)  # raises KeyError if not found

        task = Task(
            id=self._get_next_id(),
            subject=subject,
            description=description,
            blocked_by=list(blocked_by) if blocked_by else [],
            metadata=metadata or {},
        )

        # Update reverse references (blocks)
        if blocked_by:
            for dep_id in blocked_by:
                dep_task = self.get(dep_id)
                if task.id not in dep_task.blocks:
                    dep_task.blocks.append(task.id)

        self.tasks.append(task)
        return task

    def claim(self, task_id: str, owner: str) -> Task:
        """Claim a task for an agent to work on.

        Sets the task's owner and changes status to in_progress.

        Args:
            task_id: The task to claim.
            owner: Name of the agent claiming the task.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task doesn't exist.
            ValueError: If the task is not available (already claimed, blocked, or not pending).
        """
        task_id = self._normalize_id(task_id)
        task = self.get(task_id)
        if task.status != "pending":
            raise ValueError(f"Task '{task_id}' is not pending (status: {task.status})")
        if task.is_blocked:
            raise ValueError(f"Task '{task_id}' is blocked by: {task.blocked_by}")
        if task.owner is not None:
            raise ValueError(f"Task '{task_id}' is already owned by: {task.owner}")

        task.owner = owner
        task.status = "in_progress"
        task._touch()
        return task

    def complete(self, task_id: str, result: str | None = None) -> list[Task]:
        """Mark a task as completed and unblock dependent tasks.

        Args:
            task_id: The task to complete.
            result: Optional result/output of the task.

        Returns:
            List of tasks that were unblocked by this completion.

        Raises:
            KeyError: If the task doesn't exist.
            ValueError: If the task is not in_progress.
        """
        task_id = self._normalize_id(task_id)
        task = self.get(task_id)
        if task.status != "in_progress":
            raise ValueError(f"Task '{task_id}' is not in_progress (status: {task.status})")

        task.status = "completed"
        task.result = result
        task._touch()

        # Unblock dependent tasks
        unblocked = []
        for blocked_id in task.blocks:
            try:
                blocked_task = self.get(blocked_id)
                if task_id in blocked_task.blocked_by:
                    blocked_task.blocked_by.remove(task_id)
                    blocked_task._touch()
                    if not blocked_task.is_blocked and blocked_task.status == "pending":
                        unblocked.append(blocked_task)
            except KeyError:
                pass  # blocked task may have been deleted

        return unblocked

    def release(self, task_id: str) -> Task:
        """Release an in_progress task back to pending so it can be reassigned.

        Clears the owner and resets status to pending. Use this when an agent
        has stalled on a task and the leader wants to reassign it.

        Args:
            task_id: The task to release.

        Returns:
            The updated Task.

        Raises:
            KeyError: If the task doesn't exist.
            ValueError: If the task is not in_progress.
        """
        task_id = self._normalize_id(task_id)
        task = self.get(task_id)
        if task.status != "in_progress":
            raise ValueError(f"Task '{task_id}' is not in_progress (status: {task.status})")

        task.status = "pending"
        task.owner = None
        task._touch()
        return task

    def available(self) -> list[Task]:
        """Get all tasks that are available to be claimed.

        Returns tasks that are pending, not blocked, and have no owner.
        """
        return [t for t in self.tasks if t.is_available]

    def by_status(self, status: str) -> list[Task]:
        """Get all tasks with a given status."""
        return [t for t in self.tasks if t.status == status]

    def by_owner(self, owner: str) -> list[Task]:
        """Get all tasks owned by a specific agent."""
        return [t for t in self.tasks if t.owner == owner]

    def summary(self) -> str:
        """Get a human-readable summary of all tasks."""
        lines = []
        for task in self.tasks:
            status_icon = {"pending": "○", "in_progress": "◉", "completed": "✓"}[task.status]
            owner_str = f" (owner: {task.owner})" if task.owner else ""
            blocked_str = f" [blocked by {task.blocked_by}]" if task.blocked_by else ""
            lines.append(f"#{task.id} [{status_icon}] {task.subject}{owner_str}{blocked_str}")
        return "\n".join(lines)

    def to_dict_for_llm(self) -> list[dict[str, Any]]:
        """Get a simplified dict representation suitable for LLM context.

        Returns a list of task dicts with only the fields an LLM needs to see.
        """
        return [
            {
                "id": t.id,
                "subject": t.subject,
                "description": t.description,
                "status": t.status,
                "owner": t.owner,
                "blocked_by": t.blocked_by,
                "result": t.result,
            }
            for t in self.tasks
        ]
