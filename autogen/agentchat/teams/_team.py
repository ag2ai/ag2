# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Team - a named group of agents working together on tasks.

A Team manages agents, a shared task list, and an inbox messaging system.
The leader agent coordinates work by creating tasks and assigning them to agents.
Worker agents pick up tasks, do the work (via step()), and report results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from autogen.agentchat.conversable_agent import ConversableAgent

from ._step import StepResult, step
from ._task import TaskList

logger = logging.getLogger(__name__)


class Message(BaseModel):
    """A message between team members."""

    from_agent: str
    to_agent: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    message_type: Literal["text", "task_completed", "handoff"] = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Serializable configuration for recreating an agent.

    Since ConversableAgent can't be serialized directly, this stores
    the config needed to recreate one.
    """

    name: str
    system_message: str = "You are a helpful assistant."
    description: str = ""
    llm_config: dict[str, Any] = Field(default_factory=dict)
    tool_names: list[str] = Field(default_factory=list)
    is_leader: bool = False


class TeamConfig(BaseModel):
    """Serializable team configuration."""

    name: str
    description: str = ""
    agents: dict[str, AgentConfig] = Field(default_factory=dict)
    leader: str | None = None


class Team:
    """A named group of agents working together on tasks.

    The Team manages:
    - A set of agents (ConversableAgent instances)
    - A shared TaskList for work distribution
    - An inbox system for agent-to-agent messaging
    - Task tools that agents can use to interact with the task system

    Example:
        ```python
        team = Team("my-project", description="Build a website")
        team.add_agent(leader_agent, is_leader=True)
        team.add_agent(researcher_agent)
        team.add_agent(developer_agent)

        # Leader creates tasks
        result = await team.step(team.leader, "Break down: build a landing page")

        # Workers pick up and complete tasks
        result = await team.work("researcher")
        ```
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.agents: dict[str, ConversableAgent] = {}
        self.tasks = TaskList()
        self.inboxes: dict[str, list[Message]] = {}
        self._leader_name: str | None = None

    @property
    def leader(self) -> ConversableAgent | None:
        """Get the leader agent."""
        if self._leader_name:
            return self.agents.get(self._leader_name)
        return None

    def add_agent(self, agent: ConversableAgent, is_leader: bool = False) -> None:
        """Add an agent to the team.

        Registers task management tools on the agent so it can interact
        with the team's task system. Tools are only registered if the agent
        has an LLM config (tools require LLM config for schema registration).

        Args:
            agent: The ConversableAgent to add.
            is_leader: Whether this agent is the team leader.
        """
        self.agents[agent.name] = agent
        self.inboxes[agent.name] = []

        if is_leader:
            self._leader_name = agent.name

        # Only register tools if the agent has an LLM config
        if agent.llm_config:
            if is_leader:
                self._register_leader_tools(agent)
            else:
                self._register_worker_tools(agent)

    def _register_leader_tools(self, agent: ConversableAgent) -> None:
        """Register team management tools on the leader agent."""
        team = self  # capture for closures

        @agent.register_for_llm(
            description="Create a new task for the team. Use this to break down work into subtasks."
        )
        @agent.register_for_execution()
        def create_task(subject: str, description: str = "", blocked_by: str = "") -> str:
            """Create a new task. blocked_by is a comma-separated list of task IDs."""
            deps = [d.strip() for d in blocked_by.split(",") if d.strip()] if blocked_by else None
            task = team.tasks.create(subject, description, blocked_by=deps)
            return f"Created task #{task.id}: {task.subject}"

        @agent.register_for_llm(description="Get a summary of all tasks and their statuses.")
        @agent.register_for_execution()
        def list_tasks() -> str:
            """List all tasks with their current status."""
            if not team.tasks.tasks:
                return "No tasks created yet."
            return team.tasks.summary()

        @agent.register_for_llm(description="Assign a task to a team member agent.")
        @agent.register_for_execution()
        def assign_task(task_id: str, agent_name: str) -> str:
            """Assign a task to a specific agent."""
            if agent_name not in team.agents:
                return f"Error: Agent '{agent_name}' not found. Available: {list(team.agents.keys())}"
            try:
                task = team.tasks.claim(task_id, agent_name)
                return f"Assigned task #{task.id} ({task.subject}) to {agent_name}"
            except (KeyError, ValueError) as e:
                return f"Error: {e}"

        @agent.register_for_llm(
            description=(
                "Reassign a stalled or in-progress task to a different agent. "
                "Use this when an agent is stuck and another should take over."
            )
        )
        @agent.register_for_execution()
        def reassign_task(task_id: str, agent_name: str) -> str:
            """Release a task from its current owner and assign it to a new agent."""
            if agent_name not in team.agents:
                return f"Error: Agent '{agent_name}' not found. Available: {list(team.agents.keys())}"
            try:
                team.tasks.release(task_id)
                task = team.tasks.claim(task_id, agent_name)
                return f"Reassigned task #{task.id} ({task.subject}) to {agent_name}"
            except (KeyError, ValueError) as e:
                return f"Error: {e}"

        @agent.register_for_llm(description="Send a message to a team member.")
        @agent.register_for_execution()
        def send_message(to_agent: str, content: str) -> str:
            """Send a message to another agent."""
            if to_agent not in team.agents:
                return f"Error: Agent '{to_agent}' not found."
            msg = Message(from_agent=agent.name, to_agent=to_agent, content=content)
            team.inboxes[to_agent].append(msg)
            return f"Message sent to {to_agent}"

        @agent.register_for_llm(description="Get the list of team members and their descriptions.")
        @agent.register_for_execution()
        def list_team_members() -> str:
            """List all team members."""
            members = []
            for name, a in team.agents.items():
                role = " (LEADER)" if name == team._leader_name else ""
                desc = getattr(a, "description", "") or ""
                members.append(f"- {name}{role}: {desc}")
            return "\n".join(members) if members else "No team members."

    def _register_worker_tools(self, agent: ConversableAgent) -> None:
        """Register task interaction tools on a worker agent."""
        team = self
        agent_name = agent.name

        @agent.register_for_llm(description="Get your assigned tasks and any available tasks.")
        @agent.register_for_execution()
        def my_tasks() -> str:
            """Get tasks assigned to you and available tasks."""
            owned = team.tasks.by_owner(agent_name)
            available = team.tasks.available()
            lines = []
            if owned:
                lines.append("Your tasks:")
                for t in owned:
                    lines.append(f"  #{t.id} [{t.status}] {t.subject}")
            if available:
                lines.append("Available tasks:")
                for t in available:
                    lines.append(f"  #{t.id} {t.subject}: {t.description}")
            if not lines:
                lines.append("No tasks assigned or available.")
            return "\n".join(lines)

        @agent.register_for_llm(description="Claim an available task to work on it.")
        @agent.register_for_execution()
        def claim_task(task_id: str) -> str:
            """Claim an available task."""
            try:
                task = team.tasks.claim(task_id, agent_name)
                return f"Claimed task #{task.id}: {task.subject}\nDescription: {task.description}"
            except (KeyError, ValueError) as e:
                return f"Error: {e}"

        @agent.register_for_llm(description="Mark a task as completed with your result.")
        @agent.register_for_execution()
        def complete_task(task_id: str, result: str = "Done") -> str:
            """Mark a task as completed."""
            try:
                unblocked = team.tasks.complete(task_id, result)
                msg = f"Task #{task_id} completed."
                if unblocked:
                    msg += f" Unblocked: {[f'#{t.id} {t.subject}' for t in unblocked]}"
                return msg
            except (KeyError, ValueError) as e:
                return f"Error: {e}"

        @agent.register_for_llm(description="Send a message to the team leader or another agent.")
        @agent.register_for_execution()
        def send_message(to_agent: str, content: str) -> str:
            """Send a message to another agent."""
            if to_agent not in team.agents:
                return f"Error: Agent '{to_agent}' not found."
            msg = Message(from_agent=agent_name, to_agent=to_agent, content=content)
            team.inboxes[to_agent].append(msg)
            return f"Message sent to {to_agent}"

        @agent.register_for_llm(description="Check your inbox for messages from other agents.")
        @agent.register_for_execution()
        def check_inbox() -> str:
            """Check for new messages."""
            messages = team.inboxes.get(agent_name, [])
            if not messages:
                return "No new messages."
            lines = []
            for msg in messages:
                lines.append(f"From {msg.from_agent}: {msg.content}")
            team.inboxes[agent_name] = []  # clear after reading
            return "\n".join(lines)

    async def step_agent(
        self,
        agent_name: str,
        message: str,
        *,
        max_turns: int = 20,
        stream: bool = False,
        on_token: Any = None,
    ) -> StepResult:
        """Execute a step for a specific agent.

        Args:
            agent_name: Name of the agent to step.
            message: The message/instruction to send to the agent.
            max_turns: Maximum LLM call rounds.
            stream: Whether to enable streaming from the LLM.
            on_token: Callback for streaming text deltas.

        Returns:
            StepResult from the agent's step.

        Raises:
            KeyError: If the agent isn't in the team.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' not in team '{self.name}'")

        agent = self.agents[agent_name]
        return await step(agent, message, max_turns=max_turns, stream=stream, on_token=on_token)

    async def step_agent_with_context(
        self,
        agent_name: str,
        messages: list[dict[str, Any]],
        *,
        max_turns: int = 20,
    ) -> StepResult:
        """Execute a step for an agent with full conversation context.

        Args:
            agent_name: Name of the agent to step.
            messages: Full message history to pass.
            max_turns: Maximum LLM call rounds.

        Returns:
            StepResult from the agent's step.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Agent '{agent_name}' not in team '{self.name}'")

        agent = self.agents[agent_name]
        return await step(agent, messages, max_turns=max_turns)

    def send(self, from_agent: str, to_agent: str, content: str, **kwargs: Any) -> Message:
        """Send a message between agents programmatically.

        Args:
            from_agent: Sender agent name.
            to_agent: Recipient agent name.
            content: Message text.
            **kwargs: Additional keyword arguments passed to Message constructor.

        Returns:
            The Message object.
        """
        msg = Message(from_agent=from_agent, to_agent=to_agent, content=content, **kwargs)
        if to_agent in self.inboxes:
            self.inboxes[to_agent].append(msg)
        return msg

    def get_inbox(self, agent_name: str) -> list[Message]:
        """Get and clear an agent's inbox."""
        messages = list(self.inboxes.get(agent_name, []))
        self.inboxes[agent_name] = []
        return messages

    def config(self) -> TeamConfig:
        """Get a serializable configuration for this team."""
        agent_configs = {}
        for name, agent in self.agents.items():
            agent_configs[name] = AgentConfig(
                name=name,
                system_message=agent.system_message if hasattr(agent, "system_message") else "",
                description=getattr(agent, "description", "") or "",
                is_leader=(name == self._leader_name),
            )
        return TeamConfig(
            name=self.name,
            description=self.description,
            agents=agent_configs,
            leader=self._leader_name,
        )
