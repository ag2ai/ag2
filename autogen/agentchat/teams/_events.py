# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Team orchestration events.

Events emitted during team orchestration runs, compatible with AG2's
BaseEvent system. Use with `run_stream()` to drive a UI.

Event types (the `type` field on wrapped events):
    - team_task_created: A new task was created
    - team_task_assigned: A task was assigned to an agent
    - team_task_completed: A task was completed with a result
    - team_agent_step_start: An agent is about to take a turn
    - team_agent_step_complete: An agent finished a turn
    - team_handoff: An agent handed off work to another
    - team_phase: Orchestrator entered a new phase (plan/work/review/summarize)
    - team_run_complete: Orchestration finished
"""

from __future__ import annotations

from typing import Any, Literal

from autogen.events.base_event import BaseEvent, wrap_event


@wrap_event
class TeamTaskCreatedEvent(BaseEvent):
    """A task was created in the team's task list."""

    task_id: str
    subject: str
    description: str
    blocked_by: list[str]
    created_by: str  # agent name

    def print(self, f: Any = None) -> None:
        blocked = f" [blocked by {self.blocked_by}]" if self.blocked_by else ""
        (f or print)(f"[task created] #{self.task_id}: {self.subject}{blocked}")


@wrap_event
class TeamTaskAssignedEvent(BaseEvent):
    """A task was assigned to (or claimed by) an agent."""

    task_id: str
    subject: str
    agent_name: str
    assigned_by: str  # "self" if self-claimed, leader name if assigned

    def print(self, f: Any = None) -> None:
        (f or print)(f"[task assigned] #{self.task_id} → {self.agent_name}")


@wrap_event
class TeamTaskCompletedEvent(BaseEvent):
    """A task was completed with a result."""

    task_id: str
    subject: str
    agent_name: str
    result: str
    unblocked: list[str]  # task IDs that were unblocked

    def print(self, f: Any = None) -> None:
        preview = self.result[:80].replace("\n", " ")
        if len(self.result) > 80:
            preview += "..."
        unblocked_str = f" (unblocked: {self.unblocked})" if self.unblocked else ""
        (f or print)(f"[task complete] #{self.task_id}: {self.subject}{unblocked_str}\n  → {preview}")


@wrap_event
class TeamAgentStepStartEvent(BaseEvent):
    """An agent is about to take a turn."""

    agent_name: str
    task_id: str | None = None
    task_subject: str | None = None
    message_preview: str = ""

    def print(self, f: Any = None) -> None:
        task = f" (task #{self.task_id}: {self.task_subject})" if self.task_id else ""
        (f or print)(f"[step start] {self.agent_name}{task}")


@wrap_event
class TeamAgentStepCompleteEvent(BaseEvent):
    """An agent finished a turn."""

    agent_name: str
    task_id: str | None = None
    content_preview: str = ""
    content: str = ""
    tools_called: list[str] = []
    tool_call_details: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}  # {prompt_tokens, completion_tokens, total_tokens, cost, model}

    def print(self, f: Any = None) -> None:
        task = f" (task #{self.task_id})" if self.task_id else ""
        tools = f" tools={self.tools_called}" if self.tools_called else ""
        (f or print)(f"[step done] {self.agent_name}{task}{tools}")


@wrap_event
class TeamAgentStepErrorEvent(BaseEvent):
    """An agent's step failed (e.g. exceeded max turns in a tool loop)."""

    agent_name: str
    task_id: str | None = None
    task_subject: str | None = None
    error: str

    def print(self, f: Any = None) -> None:
        task = f" (task #{self.task_id})" if self.task_id else ""
        (f or print)(f"[step error] {self.agent_name}{task}: {self.error}")


@wrap_event
class TeamHandoffEvent(BaseEvent):
    """An agent handed off work to another agent."""

    from_agent: str
    to_agent: str
    message: str

    def print(self, f: Any = None) -> None:
        preview = self.message[:60].replace("\n", " ")
        (f or print)(f"[handoff] {self.from_agent} → {self.to_agent}: {preview}")


@wrap_event
class TeamPhaseEvent(BaseEvent):
    """Orchestrator entered a new phase."""

    phase: Literal["plan", "work", "review", "summarize"]
    round_number: int | None = None
    detail: str = ""

    def print(self, f: Any = None) -> None:
        round_str = f" (round {self.round_number})" if self.round_number is not None else ""
        detail_str = f": {self.detail}" if self.detail else ""
        (f or print)(f"[phase] {self.phase}{round_str}{detail_str}")


@wrap_event
class TeamRunCompleteEvent(BaseEvent):
    """Orchestration finished."""

    success: bool
    total_turns: int
    tasks_completed: int
    tasks_total: int
    summary: str

    def print(self, f: Any = None) -> None:
        status = "SUCCESS" if self.success else "INCOMPLETE"
        (f or print)(
            f"[run complete] {status} — {self.tasks_completed}/{self.tasks_total} tasks, {self.total_turns} turns"
        )
