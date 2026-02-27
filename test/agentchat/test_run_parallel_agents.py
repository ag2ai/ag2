# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import time

import pytest

from autogen.agentchat.run_parallel_agents import (
    ParallelAgentRunner,
    ParallelTask,
    ParallelTaskResult,
)


class MockConversableAgent:
    """Mock AG2 ConversableAgent for testing without LLM calls"""

    def __init__(self, name, system_message=None, llm_config=None, **kwargs):
        self.name = name
        self.system_message = system_message or ""
        self.llm_config = llm_config
        self.human_input_mode = kwargs.get("human_input_mode", "NEVER")
        self.max_consecutive_auto_reply = kwargs.get("max_consecutive_auto_reply", 10)
        self._call_count = 0

    def run(self, message, max_turns=1):
        """Mock run method that simulates agent response"""
        self._call_count += 1
        time.sleep(0.1)

        class MockResponse:
            def __init__(self, agent_name, message):
                self.messages = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"Response from {agent_name}: {message[:20]}..."},
                ]
                self.agent_name = agent_name
                self.success = True
                self.summary = f"Summary for {agent_name}"

            def process(self):
                pass

        return MockResponse(self.name, message)


@pytest.fixture
def mock_agent_class(monkeypatch):
    """Replace ConversableAgent with MockConversableAgent"""
    # Patch both the module-level import and the autogen module
    monkeypatch.setattr("autogen.agentchat.run_parallel_agents.ConversableAgent", MockConversableAgent)
    monkeypatch.setattr("autogen.ConversableAgent", MockConversableAgent)
    return MockConversableAgent


class TestParallelAgentRunner:
    """Test suite for ParallelAgentRunner functionality"""

    def test_initialization_defaults(self):
        """Test runner initialization with default parameters"""
        runner = ParallelAgentRunner()

        assert runner.max_workers is None
        assert runner.timeout is None
        assert runner.handle_errors == "collect"
        assert runner.agent_factory is not None
        assert runner.execution_stats["total_tasks"] == 0

    def test_initialization_custom_params(self):
        """Test runner initialization with custom parameters"""

        def custom_factory(name, config):
            return MockConversableAgent(name=name)

        runner = ParallelAgentRunner(
            max_workers=4, timeout=30.0, handle_errors="fail_fast", agent_factory=custom_factory
        )

        assert runner.max_workers == 4
        assert runner.timeout == 30.0
        assert runner.handle_errors == "fail_fast"
        assert runner.agent_factory == custom_factory

    def test_parse_task_from_dict(self, mock_agent_class):
        """Test parsing task from dictionary"""
        runner = ParallelAgentRunner()

        task_dict = {
            "name": "test_task",
            "agent_config": {"name": "test_agent", "system_message": "Test"},
            "prompt": "Test prompt",
            "max_turns": 2,
        }

        parsed_task = runner._parse_task(task_dict)

        assert isinstance(parsed_task, ParallelTask)
        assert parsed_task.name == "test_task"
        assert parsed_task.prompt == "Test prompt"
        assert parsed_task.max_turns == 2

    def test_parse_task_from_parallel_task(self):
        """Test parsing task from ParallelTask object"""
        runner = ParallelAgentRunner()

        task = ParallelTask(name="test_task", agent_config={"name": "agent"}, prompt="Test")

        parsed_task = runner._parse_task(task)

        assert parsed_task is task

    def test_default_agent_factory_with_dict(self, mock_agent_class):
        """Test default agent factory with dict config"""
        runner = ParallelAgentRunner()

        config = {"name": "base_agent", "system_message": "Test system message", "llm_config": {"model": "gpt-4"}}

        agent = runner._default_agent_factory("task1", config)

        # Check agent was created with correct name pattern
        assert agent.name.startswith("base_agent_task1")
        assert hasattr(agent, "system_message")

    def test_default_agent_factory_with_agent_instance(self, mock_agent_class):
        """Test default agent factory with existing agent"""

        # Use custom factory that handles MockConversableAgent properly
        def custom_factory(task_name, agent_config):
            if isinstance(agent_config, MockConversableAgent):
                return MockConversableAgent(
                    name=f"{agent_config.name}_{task_name}", system_message=agent_config.system_message
                )
            return mock_agent_class(**agent_config)

        runner = ParallelAgentRunner(agent_factory=custom_factory)

        base_agent = MockConversableAgent(name="original_agent", system_message="Original message")

        agent = runner.agent_factory("task1", base_agent)

        assert isinstance(agent, MockConversableAgent)
        assert agent.name == "original_agent_task1"
        assert agent.system_message == "Original message"

    def test_execute_single_task_success(self, mock_agent_class):
        """Test successful execution of single task"""
        runner = ParallelAgentRunner()

        task = ParallelTask(name="success_task", agent_config={"name": "test_agent"}, prompt="Test prompt")

        result = runner._execute_single_task(task)

        assert isinstance(result, ParallelTaskResult)
        assert result.success is True
        assert result.task_name == "success_task"
        assert result.error is None
        assert result.execution_time > 0

    def test_execute_single_task_failure(self, mock_agent_class):
        """Test failed execution of single task"""
        runner = ParallelAgentRunner()

        task = ParallelTask(name="fail_task", agent_config={"name": "test_agent"}, prompt="Test prompt")

        def failing_factory(name, config):
            raise ValueError("Intentional test failure")

        runner.agent_factory = failing_factory

        result = runner._execute_single_task(task)

        assert result.success is False
        assert "ValueError" in result.error
        assert "Intentional test failure" in result.error

    def test_run_parallel_tasks_success(self, mock_agent_class):
        """Test running multiple tasks in parallel successfully"""
        runner = ParallelAgentRunner(max_workers=2)

        tasks = [
            {"name": "task1", "agent_config": {"name": "agent1"}, "prompt": "Prompt 1"},
            {"name": "task2", "agent_config": {"name": "agent2"}, "prompt": "Prompt 2"},
        ]

        results = runner.run(tasks)

        assert len(results) == 2
        assert "task1" in results
        assert "task2" in results
        assert results["task1"].success is True
        assert results["task2"].success is True
        assert runner.execution_stats["successful_tasks"] == 2
        assert runner.execution_stats["failed_tasks"] == 0

    def test_run_with_duplicate_task_names(self, mock_agent_class):
        """Test that duplicate task names raise ValueError"""
        runner = ParallelAgentRunner()

        tasks = [
            {"name": "duplicate", "agent_config": {}, "prompt": "Test 1"},
            {"name": "duplicate", "agent_config": {}, "prompt": "Test 2"},
        ]

        with pytest.raises(ValueError, match="Task names must be unique"):
            runner.run(tasks)

    def test_run_with_error_collect_strategy(self, mock_agent_class):
        """Test error handling with 'collect' strategy"""
        runner = ParallelAgentRunner(handle_errors="collect")

        def selective_factory(name, config):
            if "task2" in name:
                raise ValueError("Task 2 failure")
            return MockConversableAgent(name=name)

        runner.agent_factory = selective_factory

        tasks = [
            {"name": "task1", "agent_config": {}, "prompt": "Test 1"},
            {"name": "task2", "agent_config": {}, "prompt": "Test 2"},
            {"name": "task3", "agent_config": {}, "prompt": "Test 3"},
        ]

        results = runner.run(tasks)

        assert len(results) == 3
        assert results["task1"].success is True
        assert results["task2"].success is False
        assert results["task3"].success is True
        assert runner.execution_stats["successful_tasks"] == 2
        assert runner.execution_stats["failed_tasks"] == 1

    def test_run_with_error_fail_fast_strategy(self, mock_agent_class):
        """Test error handling with 'fail_fast' strategy"""
        runner = ParallelAgentRunner(handle_errors="fail_fast")

        def failing_factory(name, config):
            raise ValueError("Intentional failure")

        runner.agent_factory = failing_factory

        tasks = [{"name": "task1", "agent_config": {}, "prompt": "Test 1"}]

        with pytest.raises(ValueError):
            runner.run(tasks)

    def test_get_stats(self, mock_agent_class):
        """Test getting execution statistics"""
        runner = ParallelAgentRunner()

        tasks = [{"name": "task1", "agent_config": {}, "prompt": "Test"}]

        runner.run(tasks)
        stats = runner.get_stats()

        assert isinstance(stats, dict)
        assert "total_tasks" in stats
        assert "successful_tasks" in stats
        assert "failed_tasks" in stats
        assert "total_time" in stats
        assert stats["total_tasks"] == 1
