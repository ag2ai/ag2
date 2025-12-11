# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import logging
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal

from dotenv import load_dotenv

from autogen import ConversableAgent

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParallelTask:
    """
    Defines a single parallel task for agent execution.
    Attributes:
        name: Unique identifier for the task
        agent_config: Configuration for creating the agent (dict or ConversableAgent instance)
        prompt: The message/prompt to send to the agent
        max_turns: Maximum conversation turns (default: 1)
        user_input: Whether to allow user input during execution (default: False for parallel)
        context: Optional additional context to pass to the agent
    """

    name: str
    agent_config: dict[str, Any] | ConversableAgent
    prompt: str
    max_turns: int = 1
    user_input: bool = False
    context: dict[str, Any] | None = None


@dataclass
class ParallelTaskResult:
    """
    Result from a parallel task execution.
    Attributes:
        task_name: Name of the completed task
        success: Whether the task completed successfully
        result: The agent's response or result object
        error: Error message if task failed
        execution_time: Time taken to execute in seconds
        agent_name: Name of the agent that executed the task
    """

    task_name: str
    success: bool
    result: Any = None
    error: str | None = None
    execution_time: float = 0.0
    agent_name: str | None = None
    summary: str | None = None


class ParallelAgentRunner:
    """
    Executes multiple AG2 agents in parallel for improved performance.
    Features:
    - Thread-based parallel execution for I/O-bound agent operations
    - Flexible error handling strategies
    - Resource management with configurable worker pools
    - Detailed execution metrics and logging
    - Backward compatible with existing AG2 agents
    Example:
        >>> runner = ParallelAgentRunner(max_workers=4)
        >>> tasks = [
        ...     {"name": "task1", "agent_config": {...}, "prompt": "Do task 1"},
        ...     {"name": "task2", "agent_config": {...}, "prompt": "Do task 2"},
        ... ]
        >>> results = runner.run(tasks)
    """

    def __init__(
        self,
        max_workers: int | None = None,
        timeout: float | None = None,
        handle_errors: Literal["collect", "fail_fast", "ignore"] = "collect",
        agent_factory: Callable | None = None,
    ):
        """
        Initialize the ParallelAgentRunner.
        Args:
            max_workers: Maximum number of parallel workers.
                        Defaults to min(32, (cpu_count + 4))
            timeout: Maximum time in seconds for each task. None for no timeout.
            handle_errors: Error handling strategy:
                - 'collect': Collect all errors and continue (default)
                - 'fail_fast': Stop on first error
                - 'ignore': Suppress errors and continue
            agent_factory: Optional custom factory function for creating agents
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.handle_errors = handle_errors
        self.agent_factory = agent_factory or self._default_agent_factory
        self.execution_stats = {"total_tasks": 0, "successful_tasks": 0, "failed_tasks": 0, "total_time": 0.0}

    def _default_agent_factory(
        self, task_name: str, agent_config: dict[str, Any] | ConversableAgent
    ) -> ConversableAgent:
        """
        Default factory for creating agent instances.

        Args:
            task_name: Name of the task (used for unique agent naming)
            agent_config: Either a dict with agent configuration or existing agent instance

        Returns:
            ConversableAgent instance
        """
        # If agent_config is already a ConversableAgent, create a copy with unique name
        if isinstance(agent_config, ConversableAgent):
            # Create new agent with same config but unique name
            return ConversableAgent(
                name=f"{agent_config.name}_{task_name}",
                system_message=agent_config.system_message,
                llm_config=agent_config.llm_config,
                human_input_mode=agent_config.human_input_mode,
                max_consecutive_auto_reply=agent_config.max_consecutive_auto_reply,
            )

        # If it's a dict, create new agent from config
        config = agent_config.copy()

        # Ensure unique agent name to avoid conflicts
        base_name = config.get("name", "agent")
        config["name"] = f"{base_name}_{task_name}_{int(time.time() * 1000)}"

        # Create the agent
        return ConversableAgent(**config)

    def _execute_single_task(self, task: ParallelTask) -> ParallelTaskResult:
        """
        Execute a single agent task.

        Args:
            task: ParallelTask to execute

        Returns:
            ParallelTaskResult with execution details
        """
        start_time = time.time()

        try:
            # Create agent instance for this task
            agent = self.agent_factory(task.name, task.agent_config)

            logger.info(f"Starting task '{task.name}' with agent '{agent.name}'")

            # Execute the agent task
            response = agent.run(message=task.prompt, max_turns=task.max_turns)
            response.process()
            # response = response
            execution_time = time.time() - start_time

            logger.info(f"Task '{task.name}' completed successfully in {execution_time:.2f}s")

            return ParallelTaskResult(
                task_name=task.name,
                success=True,
                result=response,
                execution_time=execution_time,
                agent_name=agent.name,
                summary=response.summary,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"

            logger.error(f"Task '{task.name}' failed after {execution_time:.2f}s: {error_msg}")

            return ParallelTaskResult(
                task_name=task.name, success=False, error=error_msg, execution_time=execution_time
            )

    def _parse_task(self, task_input: dict[str, Any] | ParallelTask) -> ParallelTask:
        """
        Parse task input into ParallelTask object.

        Args:
            task_input: Either a dict or ParallelTask instance

        Returns:
            ParallelTask instance
        """
        if isinstance(task_input, ParallelTask):
            return task_input

        # Convert dict to ParallelTask
        return ParallelTask(
            name=task_input["name"],
            agent_config=task_input["agent_config"],
            prompt=task_input["prompt"],
            max_turns=task_input.get("max_turns", 1),
            user_input=task_input.get("user_input", False),
            context=task_input.get("context"),
        )

    def run(self, tasks: list[dict[str, Any] | ParallelTask]) -> dict[str, ParallelTaskResult]:
        """
        Execute multiple agent tasks in parallel.

        Args:
            tasks: List of task definitions (dicts or ParallelTask instances)

        Returns:
            Dictionary mapping task names to ParallelTaskResult objects

        Raises:
            ValueError: If handle_errors='fail_fast' and a task fails

        Example:
            >>> tasks = [
            ...     {
            ...         "name": "research_task",
            ...         "agent_config": {
            ...             "name": "researcher",
            ...             "system_message": "You are a research assistant",
            ...             "llm_config": {"model": "gpt-4"},
            ...         },
            ...         "prompt": "Research AI trends",
            ...         "max_turns": 2,
            ...     }
            ... ]
            >>> results = runner.run(tasks)
        """
        start_time = time.time()

        # Parse all tasks
        parsed_tasks = [self._parse_task(t) for t in tasks]

        # Validate unique task names
        task_names = [t.name for t in parsed_tasks]
        if len(task_names) != len(set(task_names)):
            raise ValueError("Task names must be unique")

        self.execution_stats["total_tasks"] = len(parsed_tasks)

        results: dict[str, ParallelTaskResult] = {}

        logger.info(f"Starting parallel execution of {len(parsed_tasks)} tasks")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task: dict[Future, ParallelTask] = {
                executor.submit(self._execute_single_task, task): task for task in parsed_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_task, timeout=self.timeout):
                task = future_to_task[future]

                try:
                    result = future.result()
                    results[result.task_name] = result

                    if result.success:
                        self.execution_stats["successful_tasks"] += 1
                    else:
                        self.execution_stats["failed_tasks"] += 1

                        # Handle error based on strategy
                        if self.handle_errors == "fail_fast":
                            # Cancel remaining tasks
                            for f in future_to_task:
                                f.cancel()
                            raise ValueError(f"Task '{result.task_name}' failed: {result.error}")
                        elif self.handle_errors == "ignore":
                            logger.warning(f"Ignoring error in task '{result.task_name}'")

                except Exception as e:
                    error_msg = f"Unexpected error processing task '{task.name}': {str(e)}"
                    logger.error(error_msg)

                    results[task.name] = ParallelTaskResult(
                        task_name=task.name, success=False, error=error_msg, execution_time=0.0
                    )

                    self.execution_stats["failed_tasks"] += 1

                    if self.handle_errors == "fail_fast":
                        raise

        total_time = time.time() - start_time
        self.execution_stats["total_time"] = total_time

        logger.info(
            f"Parallel execution completed in {total_time:.2f}s. "
            f"Success: {self.execution_stats['successful_tasks']}, "
            f"Failed: {self.execution_stats['failed_tasks']}"
        )

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return self.execution_stats.copy()


# Backward compatibility helper function
def run_agents_parallel(
    tasks: list[dict[str, Any]], max_workers: int | None = None, timeout: float | None = None
) -> dict[str, Any]:
    """
    Simple helper function for backward compatibility.

    Args:
        tasks: List of task dictionaries
        max_workers: Maximum parallel workers
        timeout: Task timeout in seconds

    Returns:
        Dictionary of results

    Example:
        >>> results = run_agents_parallel([{"name": "task1", "agent_config": {...}, "prompt": "..."}])
    """
    runner = ParallelAgentRunner(max_workers=max_workers, timeout=timeout)
    return runner.run(tasks)
