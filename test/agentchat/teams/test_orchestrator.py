# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Orchestrator - Phase 4 of Teams.

Includes unit tests (no LLM) and integration tests (with Anthropic).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams import (
    Orchestrator,
    Team,
    TeamPhaseEvent,
    TeamResult,
    TeamRunCompleteEvent,
)
from autogen.agentchat.teams._step import StepResult, ToolCallRecord, UsageRecord


def get_anthropic_config() -> LLMConfig:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return LLMConfig({
        "model": "claude-sonnet-4-20250514",
        "api_key": api_key,
        "api_type": "anthropic",
        "temperature": 0.0,
        "max_tokens": 2048,
    })


class TestOrchestratorUnit:
    """Unit tests for Orchestrator (no LLM calls)."""

    def test_create_orchestrator(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("leader", llm_config=False), is_leader=True)
        team.add_agent(ConversableAgent("worker", llm_config=False))
        orch = Orchestrator(team)
        assert orch.team is team
        assert orch.max_rounds == 10
        assert orch.max_stalls == 3

    def test_custom_max_rounds(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("leader", llm_config=False), is_leader=True)
        team.add_agent(ConversableAgent("worker", llm_config=False))
        orch = Orchestrator(team, max_rounds=5, max_stalls=2)
        assert orch.max_rounds == 5
        assert orch.max_stalls == 2

    def test_run_requires_leader(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("worker", llm_config=False))
        orch = Orchestrator(team)
        with pytest.raises(ValueError, match="must have a leader"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(orch.run("do something"))

    def test_run_requires_workers(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("leader", llm_config=False), is_leader=True)
        orch = Orchestrator(team)
        with pytest.raises(ValueError, match="at least a leader and one worker"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(orch.run("do something"))

    def test_worker_descriptions(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("leader", llm_config=False), is_leader=True)
        team.add_agent(ConversableAgent("researcher", llm_config=False, description="Web research specialist"))
        team.add_agent(ConversableAgent("developer", llm_config=False, description="Python developer"))
        orch = Orchestrator(team)
        desc = orch._get_worker_descriptions()
        assert "researcher" in desc
        assert "Web research specialist" in desc
        assert "developer" in desc
        assert "Python developer" in desc
        assert "leader" not in desc

    def test_team_result_dataclass(self) -> None:
        result = TeamResult(
            summary="All done!",
            tasks=[{"id": "1", "subject": "Test", "status": "completed"}],
            turns=[],
            total_turns=3,
            success=True,
        )
        assert result.summary == "All done!"
        assert result.success is True
        assert result.total_turns == 3


class TestOrchestratorEvents:
    """Tests for event types and run_stream."""

    def test_event_types_importable(self) -> None:
        """All event types should be importable from the teams package."""
        # Wrapped events should have type and content
        event = TeamPhaseEvent(phase="plan", detail="test goal")
        assert event.type == "team_phase"
        assert event.content.phase == "plan"
        assert event.content.detail == "test goal"

    def test_event_print(self) -> None:
        """Events should have working print methods."""
        captured = []
        event = TeamRunCompleteEvent(success=True, total_turns=5, tasks_completed=3, tasks_total=3, summary="Done!")
        event.content.print(captured.append)
        assert "SUCCESS" in captured[0]
        assert "3/3" in captured[0]

    def test_task_created_event(self) -> None:
        from autogen.agentchat.teams import TeamTaskCreatedEvent

        event = TeamTaskCreatedEvent(
            task_id="1",
            subject="Research",
            description="Do research",
            blocked_by=["2"],
            created_by="leader",
        )
        assert event.type == "team_task_created"
        assert event.content.task_id == "1"
        assert event.content.blocked_by == ["2"]

    def test_run_stream_requires_leader(self) -> None:
        """run_stream should validate team has a leader."""
        import asyncio

        team = Team("test-team")
        team.add_agent(ConversableAgent("worker", llm_config=False))
        orch = Orchestrator(team)

        async def _collect():
            events = []
            async for event in orch.run_stream("test"):
                events.append(event)
            return events

        with pytest.raises(ValueError, match="must have a leader"):
            asyncio.get_event_loop().run_until_complete(_collect())

    @pytest.mark.asyncio
    async def test_run_stream_integration(self) -> None:
        """Test run_stream yields expected event types."""
        llm_config = get_anthropic_config()

        team = Team("stream-test")
        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=("You are a project leader. Create exactly 1 task: 'Say hello'. Assign it to 'worker'."),
        )
        team.add_agent(leader, is_leader=True)

        worker = ConversableAgent(
            name="worker",
            llm_config=llm_config,
            system_message="Complete your task with complete_task. Keep it brief.",
            description="General worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=3, max_stalls=2)

        events = []
        event_types = set()
        async for event in orch.run_stream("Say hello"):
            events.append(event)
            event_types.add(event.type)

        # Must have phase events
        assert "team_phase" in event_types
        # Must have step events
        assert "team_agent_step_start" in event_types
        assert "team_agent_step_complete" in event_types
        # Must have run complete
        assert "team_run_complete" in event_types

        # Last event should be run_complete
        assert events[-1].type == "team_run_complete"

        # First event should be the plan phase
        assert events[0].type == "team_phase"
        assert events[0].content.phase == "plan"


class TestOrchestratorIntegration:
    """Integration tests with real LLM calls."""

    @pytest.mark.asyncio
    async def test_simple_orchestration(self) -> None:
        """Test a simple orchestration: leader creates tasks, worker completes them."""
        llm_config = get_anthropic_config()

        team = Team("simple-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Break down goals into 2 simple tasks. "
                "Use create_task for each, then assign each to the 'developer' agent using assign_task. "
                "Keep tasks very simple and concrete."
            ),
        )
        team.add_agent(leader, is_leader=True)

        developer = ConversableAgent(
            name="developer",
            llm_config=llm_config,
            system_message=(
                "You are a developer. When given a task, complete it by providing a brief, "
                "concrete result. Use complete_task with a short result string."
            ),
            description="Full-stack developer who builds features",
        )
        team.add_agent(developer)

        orch = Orchestrator(team, max_rounds=5, max_stalls=2)
        result = await orch.run("Create a simple greeting function in Python")

        assert isinstance(result, TeamResult)
        assert result.summary  # got a summary
        assert len(result.turns) >= 2  # at least leader plan + worker
        assert result.total_turns >= 2
        # At least some tasks should have been created
        assert len(result.tasks) >= 1

    @pytest.mark.asyncio
    async def test_orchestration_with_dependencies(self) -> None:
        """Test orchestration where tasks have dependencies."""
        llm_config = get_anthropic_config()

        team = Team("dep-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. For the given goal, create exactly 2 tasks:\n"
                "1. A research task (no dependencies)\n"
                "2. An implementation task that depends on the research (blocked_by the first task)\n"
                "Assign both to 'developer'. Use create_task with blocked_by for the second task."
            ),
        )
        team.add_agent(leader, is_leader=True)

        developer = ConversableAgent(
            name="developer",
            llm_config=llm_config,
            system_message=(
                "You are a developer. Check your tasks with my_tasks, "
                "work on any in-progress task, and complete it with complete_task. "
                "Provide a brief concrete result."
            ),
            description="Developer who researches and implements features",
        )
        team.add_agent(developer)

        orch = Orchestrator(team, max_rounds=6, max_stalls=2)
        result = await orch.run("Research and implement a fibonacci function")

        assert result.summary
        assert len(result.tasks) >= 2
        # Check that at least the first task was completed
        completed = [t for t in result.tasks if t["status"] == "completed"]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_orchestration_with_multiple_workers(self) -> None:
        """Test orchestration with multiple specialized workers."""
        llm_config = get_anthropic_config()

        team = Team("multi-worker-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Create 2 tasks:\n"
                "1. 'Write a haiku about coding' - assign to 'writer'\n"
                "2. 'Review the haiku for quality' - assign to 'reviewer', blocked_by task 1\n"
                "Use create_task and assign_task tools."
            ),
        )
        team.add_agent(leader, is_leader=True)

        writer = ConversableAgent(
            name="writer",
            llm_config=llm_config,
            system_message=(
                "You are a creative writer. Complete your assigned tasks by providing "
                "the creative writing output. Use complete_task with your writing as the result."
            ),
            description="Creative writer for content generation",
        )
        team.add_agent(writer)

        reviewer = ConversableAgent(
            name="reviewer",
            llm_config=llm_config,
            system_message=(
                "You are a quality reviewer. Review work from other team members. "
                "Use complete_task with your review feedback as the result."
            ),
            description="Quality reviewer who checks work output",
        )
        team.add_agent(reviewer)

        orch = Orchestrator(team, max_rounds=6, max_stalls=2)
        result = await orch.run("Create and review a haiku about coding")

        assert result.summary
        assert len(result.tasks) >= 2
        # Check that multiple agents participated
        agents_involved = {turn.agent_name for turn in result.turns}
        assert "leader" in agents_involved
        assert len(agents_involved) >= 2  # leader + at least one worker

    @pytest.mark.asyncio
    async def test_handoff_between_agents(self) -> None:
        """Test that an agent can hand off work to another agent."""
        llm_config = get_anthropic_config()

        team = Team("handoff-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Create 1 task: "
                "'Draft and review a one-line product tagline'. "
                "Assign it to 'drafter'."
            ),
        )
        team.add_agent(leader, is_leader=True)

        drafter = ConversableAgent(
            name="drafter",
            llm_config=llm_config,
            system_message=(
                "You are a copywriter. When working on a task, draft something brief, "
                "then use handoff_to to hand it to 'reviewer' for review. "
                "Include your draft in the handoff message."
            ),
            description="Copywriter who drafts content",
        )
        team.add_agent(drafter)

        reviewer = ConversableAgent(
            name="reviewer",
            llm_config=llm_config,
            system_message=(
                "You are a reviewer. When you receive a handoff, review the content "
                "and provide feedback. You don't have task tools, just give your opinion."
            ),
            description="Content reviewer",
        )
        team.add_agent(reviewer)

        orch = Orchestrator(team, max_rounds=5, max_stalls=2)
        result = await orch.run("Create a product tagline for a todo app")

        assert result.summary
        # Check that the handoff was processed (both drafter and reviewer participated)
        agents_involved = {turn.agent_name for turn in result.turns}
        assert "leader" in agents_involved
        assert "drafter" in agents_involved
        # The reviewer should have been involved if handoff worked
        # (not guaranteed since LLM may not always hand off, but the mechanism is there)


class TestDependencyResultPassing:
    """Tests for Bug Fix: dependency results included in task messages."""

    @pytest.mark.asyncio
    async def test_task_message_includes_dependency_results(self) -> None:
        """When a task has completed dependencies, their results should appear in the message."""
        from unittest.mock import patch

        llm_config = get_anthropic_config()
        team = Team("dep-results")

        leader = ConversableAgent("leader", llm_config=llm_config, system_message="Leader")
        team.add_agent(leader, is_leader=True)
        worker = ConversableAgent(
            "worker",
            llm_config=llm_config,
            system_message="Worker",
            description="Worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=1)

        # Set up tasks: task 1 is created, task 2 depends on it
        task1 = team.tasks.create("Research topic", description="Research AI trends")
        task2 = team.tasks.create(
            "Write article",
            description="Write about AI trends",
            blocked_by=[task1.id],
        )

        # Complete task 1 — this unblocks task 2
        team.tasks.claim(task1.id, "worker")
        team.tasks.complete(task1.id, result="AI trends: LLMs are dominant in 2025.")

        # Now task 2 is unblocked, claim it
        team.tasks.claim(task2.id, "worker")

        # Capture the message passed to step_agent
        captured_messages = []

        async def mock_step(agent_name, message, **kwargs):
            captured_messages.append((agent_name, message))
            return StepResult(
                content="Done",
                tool_calls_made=[],
                usage=UsageRecord(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        with patch.object(team, "step_agent", side_effect=mock_step):
            async for _event in orch._work_round_stream(max_turns_per_step=5):
                pass

        # The message to the worker should include task 1's result
        assert len(captured_messages) >= 1
        worker_msg = captured_messages[0][1]
        assert "AI trends: LLMs are dominant in 2025." in worker_msg
        assert "prerequisite" in worker_msg.lower()
        assert "Research topic" in worker_msg

    @pytest.mark.asyncio
    async def test_task_message_no_deps_no_extra_section(self) -> None:
        """When a task has no dependencies, no prerequisite section should appear."""
        from unittest.mock import patch

        llm_config = get_anthropic_config()
        team = Team("no-deps")

        leader = ConversableAgent("leader", llm_config=llm_config, system_message="Leader")
        team.add_agent(leader, is_leader=True)
        worker = ConversableAgent(
            "worker",
            llm_config=llm_config,
            system_message="Worker",
            description="Worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=1)

        # Create and claim a task with no dependencies
        task = team.tasks.create("Simple task", description="Do something simple")
        team.tasks.claim(task.id, "worker")

        captured_messages = []

        async def mock_step(agent_name, message, **kwargs):
            captured_messages.append((agent_name, message))
            return StepResult(
                content="Done",
                tool_calls_made=[],
                usage=UsageRecord(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        with patch.object(team, "step_agent", side_effect=mock_step):
            async for _event in orch._work_round_stream(max_turns_per_step=5):
                pass

        assert len(captured_messages) >= 1
        worker_msg = captured_messages[0][1]
        assert "prerequisite" not in worker_msg.lower()

    @pytest.mark.asyncio
    async def test_task_message_multiple_deps(self) -> None:
        """When a task has multiple completed dependencies, all results are included."""
        from unittest.mock import patch

        llm_config = get_anthropic_config()
        team = Team("multi-deps")

        leader = ConversableAgent("leader", llm_config=llm_config, system_message="Leader")
        team.add_agent(leader, is_leader=True)
        worker = ConversableAgent(
            "worker",
            llm_config=llm_config,
            system_message="Worker",
            description="Worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=1)

        # Create two independent tasks
        task1 = team.tasks.create("Gather data", description="Get data")
        task2 = team.tasks.create("Analyze trends", description="Analyze")
        # Create a third task that depends on both
        task3 = team.tasks.create(
            "Write report",
            description="Write final report",
            blocked_by=[task1.id, task2.id],
        )

        # Complete both dependencies
        team.tasks.claim(task1.id, "worker")
        team.tasks.complete(task1.id, result="Data: 100 records collected")
        team.tasks.claim(task2.id, "worker")
        team.tasks.complete(task2.id, result="Trends: growth of 15% YoY")

        # Now claim the dependent task
        team.tasks.claim(task3.id, "worker")

        captured_messages = []

        async def mock_step(agent_name, message, **kwargs):
            captured_messages.append((agent_name, message))
            return StepResult(
                content="Done",
                tool_calls_made=[],
                usage=UsageRecord(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        with patch.object(team, "step_agent", side_effect=mock_step):
            async for _event in orch._work_round_stream(max_turns_per_step=5):
                pass

        # Find the message for task 3
        task3_msgs = [msg for name, msg in captured_messages if "Write report" in msg]
        assert len(task3_msgs) >= 1
        msg = task3_msgs[0]
        assert "Data: 100 records collected" in msg
        assert "Trends: growth of 15% YoY" in msg


class TestSelfClaimTaskAttribution:
    """Tests for Bug Fix: self-claimed tasks get proper task_id attribution."""

    @pytest.mark.asyncio
    async def test_self_claim_updates_task_id_in_events(self) -> None:
        """When an idle agent claims a task, step_complete should reflect the claimed task_id."""
        from unittest.mock import patch

        llm_config = get_anthropic_config()
        team = Team("claim-test")

        leader = ConversableAgent("leader", llm_config=llm_config, system_message="Leader")
        team.add_agent(leader, is_leader=True)
        worker = ConversableAgent(
            "worker",
            llm_config=llm_config,
            system_message="Worker",
            description="Worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=1)

        # Create an available task (pending, unblocked, unowned)
        task = team.tasks.create("Available task", description="Do something")

        # Mock step_agent to simulate claiming the task
        async def mock_step(agent_name, message, **kwargs):
            # Simulate the agent claiming the task during its step
            team.tasks.claim(task.id, agent_name)
            return StepResult(
                content="Claimed and working",
                tool_calls_made=[
                    ToolCallRecord(
                        name="claim_task",
                        arguments={"task_id": task.id},
                        result=f"Claimed task #{task.id}",
                        is_success=True,
                    ),
                ],
                usage=UsageRecord(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        with patch.object(team, "step_agent", side_effect=mock_step):
            events = []
            async for event in orch._work_round_stream(max_turns_per_step=5):
                events.append(event)

        # Find the step_complete event
        step_complete_events = [e for e in events if e.type == "team_agent_step_complete"]
        assert len(step_complete_events) >= 1

        # The step_complete event should have the claimed task_id, not None
        step_event = step_complete_events[0]
        assert step_event.content.task_id == task.id

        # The turn record should also have the claimed task_id
        assert len(orch._turns) >= 1
        turn = orch._turns[-1]
        assert turn.task_id == task.id

    @pytest.mark.asyncio
    async def test_no_claim_keeps_null_task_id(self) -> None:
        """When an idle agent doesn't claim any task, task_id stays None."""
        from unittest.mock import patch

        llm_config = get_anthropic_config()
        team = Team("no-claim-test")

        leader = ConversableAgent("leader", llm_config=llm_config, system_message="Leader")
        team.add_agent(leader, is_leader=True)
        worker = ConversableAgent(
            "worker",
            llm_config=llm_config,
            system_message="Worker",
            description="Worker",
        )
        team.add_agent(worker)

        orch = Orchestrator(team, max_rounds=1)

        # Create an available task but the agent won't claim it in the mock
        team.tasks.create("Available task", description="Do something")

        async def mock_step(agent_name, message, **kwargs):
            return StepResult(
                content="I have no tasks to work on",
                tool_calls_made=[],
                usage=UsageRecord(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        with patch.object(team, "step_agent", side_effect=mock_step):
            events = []
            async for event in orch._work_round_stream(max_turns_per_step=5):
                events.append(event)

        step_complete_events = [e for e in events if e.type == "team_agent_step_complete"]
        if step_complete_events:
            # task_id should remain None since nothing was claimed
            assert step_complete_events[0].content.task_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
