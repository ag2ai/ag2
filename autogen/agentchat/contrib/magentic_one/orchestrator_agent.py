# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import json
import traceback
from operator import le
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from autogen.agentchat import Agent, ChatResult, ConversableAgent
from autogen.logger import FileLogger

from .orchestrator_prompts import (
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_GET_FINAL_ANSWER,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    ORCHESTRATOR_SYSTEM_MESSAGE,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_UPDATE_PLAN_PROMPT,
)
from .utils import clean_and_parse_json

logger: FileLogger = FileLogger(config={})


class OrchestratorAgent(ConversableAgent):
    """OrchestratorAgent is a the lead agent of magentic onethat coordinates and directs a team of specialized agents to solve complex tasks.

    The OrchestratorAgent serves as a central coordinator, managing a team of specialized agents
    with distinct capabilities. It orchestrates task execution through a sophisticated process of:
    - Initial task planning and fact-gathering
    - Dynamic agent selection and instruction
    - Continuous progress tracking
    - Adaptive replanning to recover from errors or stalls

    Key Capabilities:
    - Directs agents specialized in web browsing, file navigation, Python code execution, and more
    - Dynamically generates and updates task plans
    - Monitors agent interactions and task progress
    - Implements intelligent recovery mechanisms when agents encounter challenges

    Core Responsibilities:
    1. Break down complex tasks into manageable subtasks
    2. Assign and direct specialized agents based on their capabilities
    3. Track and validate progress towards task completion
    4. Detect and recover from execution stalls or loops
    5. Provide a final synthesized answer or summary

    Attributes:
        _agents (List[ConversableAgent]): Specialized agents available for task execution.
        _max_rounds (int): Maximum number of interaction rounds.
        _max_stalls_before_replan (int): Threshold for detecting task progression issues.
        _max_replans (int): Maximum number of task replanning attempts.
        _return_final_answer (bool): Whether to generate a comprehensive task summary.

    Args:
        name (str): Name of the orchestrator agent.
        agents (Optional[List[ConversableAgent]]): Specialized agents to coordinate.
        max_rounds (int, optional): Maximum execution rounds. Defaults to 20.
        max_stalls_before_replan (int, optional): Stall threshold before replanning. Defaults to 3.
        max_replans (int, optional): Maximum replanning attempts. Defaults to 3.
        return_final_answer (bool, optional): Generate a final summary. Defaults to False.

        Additional arguments inherited from ConversableAgent.
    """

    DEFAULT_SYSTEM_MESSAGES = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE}]

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        system_messages: List[Dict] = DEFAULT_SYSTEM_MESSAGES,
        closed_book_prompt: str = ORCHESTRATOR_CLOSED_BOOK_PROMPT,
        plan_prompt: str = ORCHESTRATOR_PLAN_PROMPT,
        synthesize_prompt: str = ORCHESTRATOR_SYNTHESIZE_PROMPT,
        ledger_prompt: str = ORCHESTRATOR_LEDGER_PROMPT,
        update_facts_prompt: str = ORCHESTRATOR_UPDATE_FACTS_PROMPT,
        update_plan_prompt: str = ORCHESTRATOR_UPDATE_PLAN_PROMPT,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        agents: Optional[List[ConversableAgent]] = [],
        max_rounds: int = 20,
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_messages[0]["content"],
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
        )

        self._system_messages = system_messages
        self._closed_book_prompt = closed_book_prompt
        self._plan_prompt = plan_prompt
        self._synthesize_prompt = synthesize_prompt
        self._ledger_prompt = ledger_prompt
        self._update_facts_prompt = update_facts_prompt
        self._update_plan_prompt = update_plan_prompt

        if chat_messages is not None:
            # Copy existing messages into defaultdict
            for agent, messages in chat_messages.items():
                for message in messages:
                    self.send(message["content"], agent)
        self._agents = agents if agents is not None else []

        self._should_replan = True
        self._max_stalls_before_replan = max_stalls_before_replan
        self._stall_counter = 0
        self._max_replans = max_replans
        self._replan_counter = 0
        self._return_final_answer = return_final_answer
        self._max_rounds = max_rounds
        self._current_round = 0

        self._team_description = ""
        self._task = ""
        self._facts = ""
        self._plan = ""

    def broadcast_message(self, message: Dict[str, Any], sender: Optional[ConversableAgent] = None) -> None:
        """Broadcast a message to all registered agents, excluding the optional sender.

        This method sends the provided message to all agents in the orchestrator's agent list,
        with an option to exclude the sender from receiving the message.

        Args:
            message (Dict[str, Any]): The message to be broadcast to all agents.
            sender (Optional[ConversableAgent], optional): The agent to exclude from receiving the message.
                Defaults to None.
        """
        for agent in self._agents:
            if agent != sender:
                self.send(message, agent)

    def _get_plan_prompt(self, team: str) -> str:
        return self._plan_prompt.format(team=team)

    def _get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        return self._synthesize_prompt.format(
            task=task,
            team=team,
            facts=facts,
            plan=plan,
        )

    def _get_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        return self._ledger_prompt.format(task=task, team=team, names=names)

    def _get_update_facts_prompt(self, task: str, facts: str) -> str:
        return self._update_facts_prompt.format(task=task, facts=facts)

    def _get_update_plan_prompt(self, team: str) -> str:
        return self._update_plan_prompt.format(team=team)

    def _get_closed_book_prompt(self, task: str) -> str:
        return self._closed_book_prompt.format(task=task)

    def _get_team_description(self) -> str:
        """Generate a description of the team's capabilities."""
        team_description = ""
        for agent in self._agents:
            team_description += f"{agent.name}: {agent.description}\n"
        return team_description

    def _get_team_names(self) -> List[str]:
        return [agent.name for agent in self._agents]

    def _initialize_task(self, task: str) -> None:
        # called the first time a task is received
        self._task = task
        self._team_description = self._get_team_description()

        # Shallow-copy the conversation
        planning_conversation = [m for m in self._oai_messages[self]]

        # 1. GATHER FACTS
        # create a closed book task and generate a response and update the chat history
        planning_conversation.append({"role": "user", "content": self._get_closed_book_prompt(self._task)})
        is_valid_response, response = self.generate_oai_reply(self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)
        self._facts = response
        planning_conversation.append({"role": "assistant", "content": self._facts})

        # 2. CREATE A PLAN
        ## plan based on available information
        planning_conversation.append({"role": "user", "content": self._get_plan_prompt(self._team_description)})

        is_valid_response, response = self.generate_oai_reply(self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)
        self._plan = response

    def _update_facts_and_plan(self) -> None:
        # called when the orchestrator decides to replan

        planning_conversation = [m for m in self._oai_messages[self]]

        # Update the facts
        planning_conversation.append(
            {"role": "user", "content": self._get_update_facts_prompt(self._task, self._facts)}
        )

        is_valid_response, response = self.generate_oai_reply(self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)

        self._facts = response
        planning_conversation.append({"role": "assistant", "content": self._facts})

        # Update the plan
        planning_conversation.append({"role": "user", "content": self._get_update_plan_prompt(self._team_description)})

        is_valid_response, response = self.generate_oai_reply(self._system_messages + planning_conversation)

        assert is_valid_response
        assert isinstance(response, str)

        self._plan = response

    def _update_ledger(self) -> Dict[str, Any]:
        max_json_retries = 10

        team_description = self._get_team_description()
        names = self._get_team_names()
        ledger_prompt = self._get_ledger_prompt(self._task, team_description, names)

        ledger_user_messages = [{"role": "user", "content": ledger_prompt}]
        # retries in case the LLM does not return a valid JSON
        assert max_json_retries > 0
        for _ in range(max_json_retries):
            messages = self._system_messages + self._oai_messages[self] + ledger_user_messages

            is_valid_response, response = self.generate_oai_reply(
                messages,
            )

            if not is_valid_response:
                raise ValueError("No valid response generated")
            if isinstance(response, dict):
                ledger_str = response.get("content")
                if not isinstance(ledger_str, str):
                    raise ValueError(f"Expected string content, got {type(ledger_str)}")
            elif isinstance(response, str):
                ledger_str = response
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")

            try:
                ledger_dict: Dict[str, Any] = clean_and_parse_json(ledger_str)

                required_keys = [
                    "is_request_satisfied",
                    "is_in_loop",
                    "is_progress_being_made",
                    "next_speaker",
                    "instruction_or_question",
                ]
                key_error = False
                for key in required_keys:
                    if key not in ledger_dict:
                        ledger_user_messages.append({"role": "assistant", "content": ledger_str})
                        ledger_user_messages.append({"role": "user", "content": f"KeyError: '{key}'"})
                        key_error = True
                        break
                    if "answer" not in ledger_dict[key]:
                        ledger_user_messages.append({"role": "assistant", "content": ledger_str})
                        ledger_user_messages.append({"role": "user", "content": f"KeyError: '{key}.answer'"})
                        key_error = True
                        break
                if key_error:
                    continue
                return ledger_dict
            except json.JSONDecodeError as e:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={
                        "stage": "error",
                        "error_type": "JSONDecodeError",
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                        "ledger_str": ledger_str,
                    },
                )
                raise e

        raise ValueError("Failed to parse ledger information after multiple retries.")

    def _prepare_final_answer(self) -> str:
        # called when the task is complete

        final_message = {"role": "user", "content": ORCHESTRATOR_GET_FINAL_ANSWER.format(task=self._task)}
        is_valid_response, response = self.generate_oai_reply(
            self._system_messages + self._oai_messages[self] + [final_message]
        )

        assert is_valid_response
        assert isinstance(response, str)

        return response

    def _select_next_agent(self, task: dict | str) -> Optional[ConversableAgent]:
        """Select the next agent to act based on the current state."""
        taskstr: str = ""
        if isinstance(task, dict):

            if isinstance(task["content"], str):
                taskstr = task["content"]
            elif (
                isinstance(task["content"], list)
                and task["content"][0]["type"] == "text"
                and isinstance(task["content"][0]["text"], str)
            ):
                taskstr = task["content"][0]["text"]
            elif (
                isinstance(task["content"], list)
                and task["content"][1]["type"] == "text"
                and isinstance(task["content"][1]["text"], str)
            ):
                taskstr = task["content"][1]["text"]
            else:
                raise ValueError(f"Invalid task format: {task}")
        elif isinstance(task, str):
            taskstr = task

        if taskstr.strip() == "":
            return None  # Empty task

        if not self._task:
            self._initialize_task(taskstr)
            # Verify initialization
            assert len(self._task) > 0
            assert len(self._facts) > 0
            assert len(self._plan) > 0
            assert len(self._team_description) > 0

            # Create initial plan message
            synthesized_prompt = self._get_synthesize_prompt(
                self._task, self._team_description, self._facts, self._plan
            )

            # Initialize state
            self._replan_counter = 0
            self._stall_counter = 0

            # Log the initial plan
            logger.log_event(
                source=self.name, name="thought", data={"stage": "initial_plan", "plan": synthesized_prompt}
            )
            # Add to chat history
            self._append_oai_message(synthesized_prompt, "assistant", self, True)

            # Add initial plan to chat history only
            return self._select_next_agent(synthesized_prompt)

        # Orchestrate the next step
        ledger_dict = self._update_ledger()
        logger.log_event(
            source=self.name,
            name="thought",
            data={"stage": "ledger_update", "content": json.dumps(ledger_dict, indent=2)},
        )

        # Task is complete
        if ledger_dict["is_request_satisfied"]["answer"] is True:
            logger.log_event(
                source=self.name, name="thought", data={"stage": "task_complete", "message": "Request satisfied"}
            )

            if self._return_final_answer:
                # generate a final message to summarize the conversation
                final_answer = self._prepare_final_answer()
                logger.log_event(source=self.name, name="final_answer", data={"answer": final_answer})

                # Add final answer to chat history
                final_msg = {"role": "assistant", "content": final_answer}
                self._append_oai_message(final_msg, "assistant", self, True)

            return None

        # Stalled or stuck in a loop
        stalled = ledger_dict["is_in_loop"]["answer"] or not ledger_dict["is_progress_being_made"]["answer"]
        if stalled:
            self._stall_counter += 1

            # We exceeded our stall counter, so we need to replan, or exit
            if self._stall_counter > self._max_stalls_before_replan:
                self._replan_counter += 1
                self._stall_counter = 0

                # We exceeded our replan counter
                if self._replan_counter > self._max_replans:
                    logger.log_event(
                        source=self.name,
                        name="thought",
                        data={"stage": "termination", "reason": "Replan counter exceeded"},
                    )
                    return None
                # Let's create a new plan
                else:
                    logger.log_event(
                        source=self.name,
                        name="thought",
                        data={"stage": "replan", "reason": "Stalled ... Replanning .."},
                    )

                    # Update our plan.
                    self._update_facts_and_plan()

                    # Preserve initial task message
                    initial_task = self._oai_messages[self][0]

                    # Reset orchestrator history
                    self._oai_messages[self] = [initial_task]

                    # Reset all agents while preserving system messages
                    for agent in self._agents:
                        if agent._oai_system_message:
                            system_msg = agent._oai_system_message[0]
                            agent.reset()
                            agent._oai_system_message = [system_msg]

                    # Send everyone the NEW plan
                    synthesized_prompt = self._get_synthesize_prompt(
                        self._task, self._team_description, self._facts, self._plan
                    )

                    # Broadcast new plan to all agents
                    self.broadcast_message({"role": "assistant", "content": synthesized_prompt})

                    logger.log_event(
                        source=self.name, name="thought", data={"stage": "new_plan", "plan": synthesized_prompt}
                    )

                    synthesized_message = {"role": "assistant", "content": synthesized_prompt}
                    self._append_oai_message(synthesized_message, "assistant", self, True)

                    # Answer from this synthesized message
                    return self._select_next_agent(synthesized_prompt)

        # Select next agent and send instruction
        next_agent_name = ledger_dict["next_speaker"]["answer"]
        for agent in self._agents:
            if agent.name == next_agent_name:
                instruction = ledger_dict["instruction_or_question"]["answer"]

                # Log the instruction
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={"stage": f"-> {next_agent_name}", "instruction": instruction},
                )

                # Update chat history
                instruction_msg = {"role": "assistant", "content": instruction}
                self._append_oai_message(instruction_msg, "assistant", self, True)

                # Broadcast instruction to all agents
                self.broadcast_message(instruction_msg)
                return agent

        return None

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict, None]:
        """Asynchronously generate a reply by orchestrating multi-agent task execution.

        This method manages the entire lifecycle of a multi-agent task, including:
        - Initializing the task
        - Selecting and coordinating agents
        - Managing task progress and replanning
        - Handling task completion or termination

        The method follows a sophisticated orchestration process:
        1. Reset the agent states and chat histories
        2. Agent based on the differnt sub task
        3. Iteratively generate responses from agents
        4. Broadcast responses and instructions
        5. Track and manage task progress
        6. Handle replanning and stall detection
        7. Terminate when max rounds are reached or task is complete

        Args:
            messages (Optional[List[Dict[str, Any]]], optional):
                Existing message history. If None, prompts for human input.
            sender (Optional[Agent], optional):
                The sender of the initial message. Defaults to None.
            **kwargs:
                Additional keyword arguments for future extensibility.

        Returns:
            Union[str, Dict, None]:
                The final content of the last message in the conversation,
                which could be a task result, summary, or None if task failed.

        Raises:
            Various potential exceptions during agent communication and task execution.

        Notes:
            - Tracks task state through `_current_round`, `_replan_counter`, and `_stall_counter`
        """
        # Reset state
        self._current_round = 0
        self._oai_messages.clear()
        for agent in self._agents:
            agent.reset()

        if messages is None:
            message = self.get_human_input("Please provide the task: ")
        else:
            message = messages[-1]["content"]

        # Initialize the first agent selection
        next_agent = self._select_next_agent(message)

        # Continue orchestration until max rounds reached or no next agent
        while next_agent is not None and self._current_round < self._max_rounds:
            self._current_round += 1

            instructions = self._oai_messages[self][-1] if self._oai_messages[self] else None
            if not instructions:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={"stage": "error", "message": "No message found in chat history"},
                )
                break

            response = await next_agent.a_generate_reply(messages=[instructions], sender=self)

            if isinstance(response, str):
                response = response
            elif isinstance(response, dict):
                response = response["content"]
            else:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={"stage": "error", "message": f"Invalid response type: {type(response)}"},
                )
                break
            response_msg = {"role": "user", "content": response}

            # Broadcast response to all agents
            self.broadcast_message(response_msg, sender=next_agent)

            was_appended = self._append_oai_message(response_msg, "user", self, is_sending=False)
            if not was_appended:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={"stage": "error", "message": "Failed to append message to OAI messages"},
                )
                break

            next_agent = self._select_next_agent(response_msg)

            if self._current_round >= self._max_rounds:
                logger.log_event(
                    source=self.name,
                    name="thought",
                    data={"stage": "max_rounds_reached", "max_rounds": self._max_rounds},
                )

        # Track final state
        final_state = {
            "rounds_completed": self._current_round,
            "replans": self._replan_counter,
            "stalls": self._stall_counter,
            "task_completed": next_agent is None and self._current_round < self._max_rounds,
        }
        logger.log_event(source=self.name, name="thought", data={"stage": "final_state", "state": final_state})

        # Return chat result with all relevant info
        return self._oai_messages[self][-1]["content"]

    async def a_initiate_chats(self, chat_queue: List[Dict[str, Any]]) -> Dict[int, ChatResult]:
        raise NotImplementedError

    def initiate_chats(self, chat_queue: List[Dict[str, Any]]) -> List[ChatResult]:
        raise NotImplementedError
