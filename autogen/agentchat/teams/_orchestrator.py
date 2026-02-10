# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Team Orchestrator - drives a team through task decomposition, assignment, and execution.

The orchestrator takes a high-level task, uses a leader agent to decompose it,
assigns work to team members, runs agents to completion, and supports handoffs
between agents.

Supports two consumption modes:
    # Batch: get the final result
    result = await orchestrator.run("Build a landing page")

    # Streaming: iterate events as they happen (for UI)
    async for event in orchestrator.run_stream("Build a landing page"):
        event.print()
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from autogen.agentchat.conversable_agent import ConversableAgent

from ._events import (
    TeamAgentStepCompleteEvent,
    TeamAgentStepErrorEvent,
    TeamAgentStepStartEvent,
    TeamHandoffEvent,
    TeamPhaseEvent,
    TeamRunCompleteEvent,
    TeamTaskAssignedEvent,
    TeamTaskCompletedEvent,
    TeamTaskCreatedEvent,
)
from ._step import StepResult, ToolCallRecord, UsageRecord
from ._team import Team

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnRecord:
    """Record of a single agent's turn during orchestration."""

    agent_name: str
    task_id: str | None
    step_result: StepResult
    handoff_to: str | None = None
    handoff_message: str | None = None


@dataclass
class TeamResult:
    """Result of a full orchestration run.

    Attributes:
        summary: Final summary text from the leader agent.
        tasks: Snapshot of all tasks at completion.
        turns: Record of every agent turn that happened.
        total_turns: Number of agent turns executed.
        success: Whether all tasks were completed.
    """

    summary: str
    tasks: list[dict[str, Any]] = field(default_factory=list)
    turns: list[AgentTurnRecord] = field(default_factory=list)
    total_turns: int = 0
    success: bool = False


class Orchestrator:
    """Drives a team through task decomposition, assignment, and execution.

    The orchestrator manages the full lifecycle:
    1. Leader decomposes a high-level goal into tasks
    2. Leader assigns tasks to agents (or agents claim them)
    3. Agents work on their tasks via step()
    4. Agents can hand off to other agents
    5. Leader monitors progress and coordinates
    6. Returns combined results when all tasks complete

    Example (batch):
        ```python
        orchestrator = Orchestrator(team)
        result = await orchestrator.run("Build a landing page")
        print(result.summary)
        ```

    Example (streaming for UI):
        ```python
        async for event in orchestrator.run_stream("Build a landing page"):
            if event.type == "team_task_created":
                print(f"New task: {event.content.subject}")
            elif event.type == "team_task_completed":
                print(f"Done: {event.content.subject}")
        ```
    """

    def __init__(
        self,
        team: Team,
        *,
        max_rounds: int = 10,
        max_stalls: int = 3,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            team: The Team to orchestrate. Must have a leader and at least one worker.
            max_rounds: Maximum orchestration rounds (plan-work-review cycles).
            max_stalls: Maximum consecutive rounds with no progress before stopping.
        """
        self.team = team
        self.max_rounds = max_rounds
        self.max_stalls = max_stalls
        self._turns: list[AgentTurnRecord] = []
        self._handoffs: list[tuple[str, str]] = []  # (target_agent, message) queue
        self._register_handoff_tools()

    def _register_handoff_tools(self) -> None:
        """Register handoff tools on all agents that have LLM config."""
        team = self.team

        for agent_name, agent in team.agents.items():
            if not agent.llm_config:
                continue
            if agent_name == team._leader_name:
                continue  # leader doesn't hand off

            self._register_handoff_for_agent(agent, agent_name)

    def _register_handoff_for_agent(self, agent: ConversableAgent, agent_name: str) -> None:
        """Register handoff tool for a specific agent."""
        orchestrator = self

        @agent.register_for_llm(
            description=(
                "Hand off work to another team member. Use this when a task requires "
                "a different specialist. Your turn ends and the target agent continues."
            )
        )
        @agent.register_for_execution()
        def handoff_to(target_agent: str, message: str) -> str:
            """Hand off to another agent with a message about what to do."""
            if target_agent not in orchestrator.team.agents:
                available = [n for n in orchestrator.team.agents if n != agent_name]
                return f"Error: Agent '{target_agent}' not found. Available: {available}"
            if target_agent == agent_name:
                return "Error: Cannot hand off to yourself."

            orchestrator._handoffs.append((target_agent, message))
            orchestrator.team.send(agent_name, target_agent, message, message_type="handoff")
            return f"Handing off to {target_agent}. Your turn will end after this response."

    async def run(
        self,
        goal: str,
        *,
        max_turns_per_step: int = 20,
    ) -> TeamResult:
        """Run the full orchestration loop for a goal (batch mode).

        Collects all events internally and returns the final TeamResult.

        Args:
            goal: The high-level task/goal to accomplish.
            max_turns_per_step: Maximum LLM turns per individual agent step.

        Returns:
            TeamResult with summary, task snapshots, and turn records.
        """
        # Consume all events, return the final result
        result = None
        async for event in self.run_stream(goal, max_turns_per_step=max_turns_per_step):
            if event.type == "team_run_complete":
                result = event

        if result is None:
            return TeamResult(summary="No events produced.", success=False)

        return TeamResult(
            summary=result.content.summary,
            tasks=self.team.tasks.to_dict_for_llm(),
            turns=list(self._turns),
            total_turns=len(self._turns),
            success=result.content.success,
        )

    async def run_stream(
        self,
        goal: str,
        *,
        max_turns_per_step: int = 20,
        stream: bool = False,
        on_token: Any = None,
    ) -> AsyncIterator:
        """Run orchestration as an async generator, yielding events as they happen.

        This is the primary method for driving a UI. Events are yielded as
        wrapped AG2 BaseEvent objects with a `type` field and `content` field.

        Event types:
            - team_phase: Orchestrator entered a new phase
            - team_task_created: A task was created
            - team_task_assigned: A task was assigned
            - team_task_completed: A task was completed
            - team_agent_step_start: An agent started a turn
            - team_agent_step_complete: An agent finished a turn
            - team_handoff: An agent handed off to another
            - team_run_complete: Run finished

        Args:
            goal: The high-level task/goal to accomplish.
            max_turns_per_step: Maximum LLM turns per individual agent step.
            stream: Whether to enable LLM streaming for real-time token output.
            on_token: Callback invoked with each text delta when streaming.

        Yields:
            Wrapped BaseEvent objects.
        """
        if self.team.leader is None:
            raise ValueError("Team must have a leader agent to run orchestration.")

        if len(self.team.agents) < 2:
            raise ValueError("Team must have at least a leader and one worker.")

        self._turns = []
        self._handoffs = []

        # Snapshot task count before planning to detect new tasks
        tasks_before = len(self.team.tasks.tasks)

        # --- Phase 1: Plan ---
        yield TeamPhaseEvent(phase="plan", detail=goal)

        leader = self.team.leader
        worker_info = self._get_worker_descriptions()

        plan_message = (
            f"Goal: {goal}\n\n"
            f"Team members available:\n{worker_info}\n\n"
            "Break this goal into concrete tasks using create_task. "
            "Then assign each task to the best team member using assign_task. "
            "Set up any dependencies between tasks (blocked_by) where order matters."
        )

        yield TeamAgentStepStartEvent(agent_name=leader.name, message_preview=plan_message[:100])

        plan_result = await self.team.step_agent(
            leader.name,
            plan_message,
            max_turns=max_turns_per_step,
            stream=stream,
            on_token=on_token,
        )
        self._turns.append(AgentTurnRecord(agent_name=leader.name, task_id=None, step_result=plan_result))

        yield TeamAgentStepCompleteEvent(
            agent_name=leader.name,
            content_preview=plan_result.content[:120] if plan_result.content else "",
            content=plan_result.content,
            tools_called=[tc.name for tc in plan_result.tool_calls_made],
            tool_call_details=self._serialize_tool_calls(plan_result.tool_calls_made),
            usage=self._serialize_usage(plan_result.usage),
        )

        # Emit events for tasks created and assigned during planning
        for event in self._emit_task_events(tasks_before, leader.name, plan_result.tool_calls_made):
            yield event

        if not self.team.tasks.tasks:
            yield TeamRunCompleteEvent(
                success=False,
                total_turns=len(self._turns),
                tasks_completed=0,
                tasks_total=0,
                summary=plan_result.content,
            )
            return

        # --- Phase 2: Work loop ---
        stall_count = 0
        for round_num in range(self.max_rounds):
            yield TeamPhaseEvent(phase="work", round_number=round_num + 1)

            progress_made = False
            async for event in self._work_round_stream(
                max_turns_per_step,
                stream=stream,
                on_token=on_token,
            ):
                yield event
                # Track progress from task completions
                if event.type == "team_task_completed" or event.type == "team_task_assigned":
                    progress_made = True

            all_done = all(t.status == "completed" for t in self.team.tasks.tasks)
            if all_done:
                break

            if progress_made:
                stall_count = 0
            else:
                stall_count += 1
                if stall_count >= self.max_stalls:
                    logger.warning(f"Orchestrator stopping after {stall_count} rounds with no progress.")
                    break

                yield TeamPhaseEvent(phase="review", round_number=round_num + 1)
                async for event in self._leader_review_stream(
                    max_turns_per_step,
                    stream=stream,
                    on_token=on_token,
                ):
                    yield event

        # --- Phase 3: Summarize ---
        yield TeamPhaseEvent(phase="summarize")
        summary = await self._leader_summarize(
            goal,
            max_turns_per_step,
            stream=stream,
            on_token=on_token,
        )

        all_done = all(t.status == "completed" for t in self.team.tasks.tasks)
        completed_count = sum(1 for t in self.team.tasks.tasks if t.status == "completed")

        yield TeamRunCompleteEvent(
            success=all_done,
            total_turns=len(self._turns),
            tasks_completed=completed_count,
            tasks_total=len(self.team.tasks.tasks),
            summary=summary,
        )

    def _emit_task_events(
        self,
        tasks_before: int,
        actor: str,
        tool_calls: list[ToolCallRecord],
    ) -> list:
        """Generate events for tasks created/assigned since tasks_before index."""
        events = []
        # Emit for newly created tasks
        for task in self.team.tasks.tasks[tasks_before:]:
            events.append(
                TeamTaskCreatedEvent(
                    task_id=task.id,
                    subject=task.subject,
                    description=task.description,
                    blocked_by=task.blocked_by,
                    created_by=actor,
                )
            )

        # Emit assign events from tool calls
        for tc in tool_calls:
            if tc.name == "assign_task" and tc.is_success and not tc.result.startswith("Error"):
                agent_name = tc.arguments.get("agent_name", "")
                task_id = tc.arguments.get("task_id", "")
                task_id = self.team.tasks._normalize_id(task_id)
                try:
                    task = self.team.tasks.get(task_id)
                    events.append(
                        TeamTaskAssignedEvent(
                            task_id=task_id,
                            subject=task.subject,
                            agent_name=agent_name,
                            assigned_by=actor,
                        )
                    )
                except KeyError:
                    pass

        return events

    def _emit_step_task_events(
        self,
        agent_name: str,
        result: StepResult,
        tasks_before: int,
    ) -> list:
        """Emit task events that happened during an agent step."""
        events = []

        # New tasks created by this agent
        events.extend(self._emit_task_events(tasks_before, agent_name, result.tool_calls_made))

        # Task completions
        for tc in result.tool_calls_made:
            if tc.name == "complete_task" and tc.is_success and not tc.result.startswith("Error"):
                task_id = self.team.tasks._normalize_id(tc.arguments.get("task_id", ""))
                try:
                    task = self.team.tasks.get(task_id)
                    # Find unblocked tasks from the result string
                    unblocked = []
                    if "Unblocked" in tc.result:
                        for t in self.team.tasks.tasks:
                            if t.is_available and t.id != task_id:
                                unblocked.append(t.id)

                    events.append(
                        TeamTaskCompletedEvent(
                            task_id=task_id,
                            subject=task.subject,
                            agent_name=agent_name,
                            result=task.result or tc.arguments.get("result", ""),
                            unblocked=unblocked,
                        )
                    )
                except KeyError:
                    pass

            # Task claims (self-claim)
            if tc.name == "claim_task" and tc.is_success and not tc.result.startswith("Error"):
                task_id = self.team.tasks._normalize_id(tc.arguments.get("task_id", ""))
                try:
                    task = self.team.tasks.get(task_id)
                    events.append(
                        TeamTaskAssignedEvent(
                            task_id=task_id,
                            subject=task.subject,
                            agent_name=agent_name,
                            assigned_by="self",
                        )
                    )
                except KeyError:
                    pass

            # Handoffs
            if tc.name == "handoff_to" and tc.is_success and not tc.result.startswith("Error"):
                target = tc.arguments.get("target_agent", "")
                message = tc.arguments.get("message", "")
                events.append(
                    TeamHandoffEvent(
                        from_agent=agent_name,
                        to_agent=target,
                        message=message,
                    )
                )

        return events

    async def _step_agent_with_events(
        self,
        agent_name: str,
        message: str,
        *,
        task_id: str | None = None,
        task_subject: str | None = None,
        max_turns: int = 20,
        stream: bool = False,
        on_token: Any = None,
    ) -> AsyncIterator:
        """Step an agent and yield events before/after.

        If the agent exceeds max_turns (tool loop) or hits another runtime
        error, a ``TeamAgentStepErrorEvent`` is yielded instead of crashing.
        The task stays in-progress so the leader can reassign it on the next
        review cycle.
        """
        yield TeamAgentStepStartEvent(
            agent_name=agent_name,
            task_id=task_id,
            task_subject=task_subject,
            message_preview=message[:100],
        )

        tasks_before = len(self.team.tasks.tasks)
        try:
            result = await self.team.step_agent(
                agent_name,
                message,
                max_turns=max_turns,
                stream=stream,
                on_token=on_token,
            )
        except (RuntimeError, ValueError) as exc:
            logger.warning(f"Agent '{agent_name}' step failed: {exc}")
            yield TeamAgentStepErrorEvent(
                agent_name=agent_name,
                task_id=task_id,
                task_subject=task_subject,
                error=str(exc),
            )
            return

        # If no task_id was provided, check if the agent claimed one during this step
        actual_task_id = task_id
        if task_id is None:
            for tc in result.tool_calls_made:
                if tc.name == "claim_task" and tc.is_success and not tc.result.startswith("Error"):
                    actual_task_id = self.team.tasks._normalize_id(tc.arguments.get("task_id", ""))
                    with contextlib.suppress(KeyError):
                        self.team.tasks.get(actual_task_id)
                    break

        self._turns.append(
            AgentTurnRecord(
                agent_name=agent_name,
                task_id=actual_task_id,
                step_result=result,
            )
        )

        yield TeamAgentStepCompleteEvent(
            agent_name=agent_name,
            task_id=actual_task_id,
            content_preview=result.content[:120] if result.content else "",
            content=result.content,
            tools_called=[tc.name for tc in result.tool_calls_made],
            tool_call_details=self._serialize_tool_calls(result.tool_calls_made),
            usage=self._serialize_usage(result.usage),
        )

        # Emit task-related events from this step
        for event in self._emit_step_task_events(agent_name, result, tasks_before):
            yield event

    async def _work_round_stream(
        self,
        max_turns_per_step: int,
        *,
        stream: bool = False,
        on_token: Any = None,
    ) -> AsyncIterator:
        """Execute one work round, yielding events."""

        # Process handoffs first
        while self._handoffs:
            target_agent, message = self._handoffs.pop(0)
            if target_agent in self.team.agents and target_agent != self.team._leader_name:
                async for event in self._step_agent_with_events(
                    target_agent,
                    message,
                    max_turns=max_turns_per_step,
                    stream=stream,
                    on_token=on_token,
                ):
                    yield event

        # Step agents with in_progress tasks
        for agent_name, agent in self.team.agents.items():
            if agent_name == self.team._leader_name:
                continue

            agent_tasks = self.team.tasks.by_owner(agent_name)
            in_progress = [t for t in agent_tasks if t.status == "in_progress"]

            for task in in_progress:
                task_message = (
                    f"You are working on task #{task.id}: {task.subject}\nDescription: {task.description}\n\n"
                )

                # Include results from completed dependencies so the agent has context
                dep_results = []
                for other_task in self.team.tasks.tasks:
                    if other_task.status == "completed" and task.id in other_task.blocks and other_task.result:
                        dep_results.append(
                            f"Result from task #{other_task.id} ({other_task.subject}):\n{other_task.result}"
                        )
                if dep_results:
                    task_message += (
                        "--- Results from prerequisite tasks ---\n"
                        + "\n\n".join(dep_results)
                        + "\n--- End prerequisite results ---\n\n"
                    )

                task_message += (
                    "Complete this task using your available tools. "
                    "When done, use complete_task to mark it finished with your result. "
                    "If you need help from another specialist, use handoff_to."
                )

                inbox = self.team.inboxes.get(agent_name, [])
                if inbox:
                    inbox_text = "\n".join(f"Message from {m.from_agent}: {m.content}" for m in inbox)
                    task_message += f"\n\nMessages in your inbox:\n{inbox_text}"
                    self.team.inboxes[agent_name] = []

                async for event in self._step_agent_with_events(
                    agent_name,
                    task_message,
                    task_id=task.id,
                    task_subject=task.subject,
                    max_turns=max_turns_per_step,
                    stream=stream,
                    on_token=on_token,
                ):
                    yield event

        # Idle agents claim available tasks
        available = self.team.tasks.available()
        if available:
            for agent_name, agent in self.team.agents.items():
                if agent_name == self.team._leader_name:
                    continue
                if not agent.llm_config:
                    continue

                agent_in_progress = [t for t in self.team.tasks.by_owner(agent_name) if t.status == "in_progress"]
                if agent_in_progress:
                    continue

                available = self.team.tasks.available()
                if not available:
                    break

                claim_message = (
                    "You have no current tasks. Check available tasks with my_tasks "
                    "and claim one that matches your skills. Then work on it."
                )
                async for event in self._step_agent_with_events(
                    agent_name,
                    claim_message,
                    max_turns=max_turns_per_step,
                    stream=stream,
                    on_token=on_token,
                ):
                    yield event

    async def _leader_review_stream(
        self,
        max_turns_per_step: int,
        *,
        stream: bool = False,
        on_token: Any = None,
    ) -> AsyncIterator:
        """Leader reviews progress, yielding events."""
        leader = self.team.leader
        if leader is None:
            return

        review_message = (
            "Review the current task status using list_tasks. "
            "Check if any tasks need reassignment or if there are blockers. "
            "Take action to keep things moving: reassign stalled tasks, "
            "create additional tasks if needed, or send messages to agents."
        )

        async for event in self._step_agent_with_events(
            leader.name,
            review_message,
            max_turns=max_turns_per_step,
            stream=stream,
            on_token=on_token,
        ):
            yield event

    async def _leader_summarize(
        self,
        goal: str,
        max_turns_per_step: int,
        *,
        stream: bool = False,
        on_token: Any = None,
    ) -> str:
        """Have the leader produce a final summary."""
        leader = self.team.leader
        if leader is None:
            return "No leader to summarize."

        task_summary = self.team.tasks.summary()
        results = []
        for task in self.team.tasks.tasks:
            if task.result:
                results.append(f"Task #{task.id} ({task.subject}): {task.result}")

        results_text = "\n".join(results) if results else "No task results yet."

        summary_message = (
            f"The goal was: {goal}\n\n"
            f"Task summary:\n{task_summary}\n\n"
            f"Task results:\n{results_text}\n\n"
            "Provide a concise final summary of what was accomplished. "
            "Be specific about the results from each task."
        )

        result = await self.team.step_agent(
            leader.name,
            summary_message,
            max_turns=max_turns_per_step,
            stream=stream,
            on_token=on_token,
        )
        self._turns.append(AgentTurnRecord(agent_name=leader.name, task_id=None, step_result=result))
        return result.content

    @staticmethod
    def _serialize_usage(usage: UsageRecord) -> dict:
        """Serialize a UsageRecord to a plain dict for event payloads."""
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "cost": usage.cost,
            "model": usage.model,
        }

    @staticmethod
    def _serialize_tool_calls(tool_calls: list[ToolCallRecord]) -> list[dict]:
        """Serialize tool call records to plain dicts for event payloads."""
        return [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "is_success": tc.is_success,
            }
            for tc in tool_calls
        ]

    def _get_worker_descriptions(self) -> str:
        """Get formatted descriptions of all worker agents."""
        lines = []
        for name, agent in self.team.agents.items():
            if name == self.team._leader_name:
                continue
            desc = getattr(agent, "description", "") or "General-purpose agent"
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines) if lines else "No workers available."
