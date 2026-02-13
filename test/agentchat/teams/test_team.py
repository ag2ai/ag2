# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Team class - Phase 3 of Teams.

Includes both unit tests (no LLM) and integration tests (with Anthropic).
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv(os.path.expanduser("~/.env"))

from autogen import ConversableAgent, LLMConfig
from autogen.agentchat.teams import Message, Team


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


class TestTeamUnit:
    """Unit tests for Team (no LLM calls)."""

    def test_create_team(self) -> None:
        team = Team("test-team", description="A test team")
        assert team.name == "test-team"
        assert team.description == "A test team"
        assert len(team.agents) == 0

    def test_add_agent(self) -> None:
        team = Team("test-team")
        agent = ConversableAgent("worker-1", llm_config=False)
        team.add_agent(agent)
        assert "worker-1" in team.agents
        assert "worker-1" in team.inboxes

    def test_add_leader(self) -> None:
        team = Team("test-team")
        leader = ConversableAgent("leader", llm_config=False)
        team.add_agent(leader, is_leader=True)
        assert team.leader is leader
        assert team._leader_name == "leader"

    def test_messaging(self) -> None:
        team = Team("test-team")
        team.add_agent(ConversableAgent("a", llm_config=False))
        team.add_agent(ConversableAgent("b", llm_config=False))

        msg = team.send("a", "b", "Hello from A!")
        assert isinstance(msg, Message)
        assert msg.from_agent == "a"
        assert msg.to_agent == "b"

        inbox = team.get_inbox("b")
        assert len(inbox) == 1
        assert inbox[0].content == "Hello from A!"

        # Inbox should be cleared after reading
        inbox2 = team.get_inbox("b")
        assert len(inbox2) == 0

    def test_config_serialization(self) -> None:
        team = Team("my-team", description="Test")
        team.add_agent(ConversableAgent("leader", llm_config=False), is_leader=True)
        team.add_agent(ConversableAgent("worker", llm_config=False))

        config = team.config()
        assert config.name == "my-team"
        assert config.leader == "leader"
        assert "leader" in config.agents
        assert "worker" in config.agents
        assert config.agents["leader"].is_leader is True

        # Serialize and verify
        json_str = config.model_dump_json()
        assert "my-team" in json_str

    def test_step_unknown_agent_raises(self) -> None:
        team = Team("test")
        with pytest.raises(KeyError, match="not in team"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(team.step_agent("unknown", "hello"))


class TestTeamIntegration:
    """Integration tests with real LLM calls."""

    @pytest.mark.asyncio
    async def test_leader_creates_tasks(self) -> None:
        """Test that the leader can create tasks using its tools."""
        llm_config = get_anthropic_config()

        team = Team("website-project", description="Build a landing page")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Break down work into tasks using create_task. "
                "Create 3 simple tasks for building a landing page. "
                "Use the create_task tool for each one. Be concise."
            ),
        )
        team.add_agent(leader, is_leader=True)

        worker = ConversableAgent(
            name="developer",
            llm_config=llm_config,
            system_message="You are a developer.",
            description="Frontend developer for HTML/CSS/JS",
        )
        team.add_agent(worker)

        result = await team.step_agent(
            "leader",
            "Create tasks for building a simple landing page. Make exactly 3 tasks.",
        )

        assert result.content  # got a response
        assert len(result.tool_calls_made) >= 3  # created at least 3 tasks
        assert len(team.tasks.tasks) >= 3
        assert all(tc.name == "create_task" for tc in result.tool_calls_made if tc.name == "create_task")

    @pytest.mark.asyncio
    async def test_worker_claims_and_completes_task(self) -> None:
        """Test that a worker can claim and complete a task."""
        llm_config = get_anthropic_config()

        team = Team("test-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message="You are a project leader.",
        )
        team.add_agent(leader, is_leader=True)

        worker = ConversableAgent(
            name="developer",
            llm_config=llm_config,
            system_message=(
                "You are a developer. Check your tasks, claim an available one, "
                "and immediately complete it with a brief result. Use the tools provided."
            ),
        )
        team.add_agent(worker)

        # Pre-create a task
        team.tasks.create("Write hello world", "Create a simple hello world HTML page")

        result = await team.step_agent(
            "developer",
            "Check available tasks, claim one, and complete it with a brief result.",
        )

        assert result.content
        tool_names = [tc.name for tc in result.tool_calls_made]
        assert "my_tasks" in tool_names or "claim_task" in tool_names
        # Task should be claimed and/or completed
        task = team.tasks.get("1")
        assert task.owner == "developer" or task.status == "completed"

    @pytest.mark.asyncio
    async def test_full_leader_worker_flow(self) -> None:
        """Test a complete flow: leader creates tasks, worker picks up and completes."""
        llm_config = get_anthropic_config()

        team = Team("quick-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Use create_task to create tasks. "
                "After creating tasks, use list_tasks to verify. Be concise."
            ),
        )
        team.add_agent(leader, is_leader=True)

        worker = ConversableAgent(
            name="worker",
            llm_config=llm_config,
            system_message=(
                "You are a worker. Use my_tasks to see available work, "
                "claim_task to claim one, and complete_task to finish it. "
                "Complete the task with a brief result. Be concise."
            ),
        )
        team.add_agent(worker)

        # Step 1: Leader creates a task
        await team.step_agent(
            "leader",
            "Create one task: 'Write a greeting function'. Then list all tasks.",
        )
        assert len(team.tasks.tasks) >= 1

        # Step 2: Worker claims and completes
        await team.step_agent(
            "worker",
            "Check available tasks, claim one, and complete it with: 'def greet(): return \"Hello!\"'",
        )

        # Verify the task was completed
        task = team.tasks.get("1")
        assert task.status == "completed"
        assert task.owner == "worker"

    @pytest.mark.asyncio
    async def test_leader_reassigns_stalled_task(self) -> None:
        """Test that the leader can reassign an in_progress task to a different agent."""
        llm_config = get_anthropic_config()

        team = Team("reassign-project")

        leader = ConversableAgent(
            name="leader",
            llm_config=llm_config,
            system_message=(
                "You are a project leader. Use reassign_task to reassign stalled tasks to different agents. Be concise."
            ),
        )
        team.add_agent(leader, is_leader=True)

        worker1 = ConversableAgent(
            name="worker-1",
            llm_config=llm_config,
            system_message="You are a worker.",
            description="General worker",
        )
        team.add_agent(worker1)

        worker2 = ConversableAgent(
            name="worker-2",
            llm_config=llm_config,
            system_message="You are a worker.",
            description="General worker",
        )
        team.add_agent(worker2)

        # Pre-create a task and assign it to worker-1 (simulating a stalled task)
        team.tasks.create("Write tests", "Write unit tests for the project")
        team.tasks.claim("1", "worker-1")
        assert team.tasks.get("1").owner == "worker-1"
        assert team.tasks.get("1").status == "in_progress"

        # Leader reassigns the stalled task
        result = await team.step_agent(
            "leader",
            "Task #1 is stalled. Reassign task #1 to worker-2.",
        )

        task = team.tasks.get("1")
        assert task.owner == "worker-2"
        assert task.status == "in_progress"
        tool_names = [tc.name for tc in result.tool_calls_made]
        assert "reassign_task" in tool_names


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
