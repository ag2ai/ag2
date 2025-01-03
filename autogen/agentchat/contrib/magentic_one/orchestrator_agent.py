# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from autogen.agentchat import Agent, ConversableAgent


class OrchestratorAgent(ConversableAgent):
    """
    A class that manages the orchestration of multiple agents to complete a task.
    """

    DEFAULT_SYSTEM_MESSAGES = [{"role": "system", "content": "System message for the orchestrator agent."}]

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
        closed_book_prompt: str = "Prompt for gathering facts before planning.",
        plan_prompt: str = "Prompt for creating a plan.",
        synthesize_prompt: str = "Prompt for synthesizing information.",
        ledger_prompt: str = "Prompt for updating the ledger.",
        update_facts_prompt: str = "Prompt for updating facts.",
        update_plan_prompt: str = "Prompt for updating the plan.",
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        agents: Optional[List[ConversableAgent]] = [],
        max_rounds: int = 20,
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
        **kwargs,
    ):
        """
        Initializes the OrchestratorAgent.

        Args:
            name (str): The name of the agent.
            is_termination_msg (Optional[Callable[[Dict], bool]]): A function to determine if a message is a termination message.
            max_consecutive_auto_reply (Optional[int]): The maximum number of consecutive auto replies.
            human_input_mode (Literal["ALWAYS", "NEVER", "TERMINATE"]): The human input mode.
            function_map (Optional[Dict[str, Callable]]): A map of function names to functions.
            code_execution_config (Union[Dict, Literal[False]]): The code execution configuration.
            llm_config (Optional[Union[Dict, Literal[False]]]): The LLM configuration.
            default_auto_reply (Union[str, Dict]): The default auto reply.
            description (Optional[str]): The description of the agent.
            system_messages (List[Dict]): The system messages.
            closed_book_prompt (str): The prompt for gathering facts before planning.
            plan_prompt (str): The prompt for creating a plan.
            synthesize_prompt (str): The prompt for synthesizing information.
            ledger_prompt (str): The prompt for updating the ledger.
            update_facts_prompt (str): The prompt for updating facts.
            update_plan_prompt (str): The prompt for updating the plan.
            chat_messages (Optional[Dict[Agent, List[Dict]]]): Initial chat messages.
            silent (Optional[bool]): Whether to suppress print statements.
            agents (Optional[List[ConversableAgent]]): The list of agents to orchestrate.
            max_rounds (int): The maximum number of rounds.
            max_stalls_before_replan (int): The maximum number of stalls before replanning.
            max_replans (int): The maximum number of replans.
            return_final_answer (bool): Whether to return the final answer.
            **kwargs: Additional keyword arguments.
        """
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
        self._ledger = {}

    def broadcast_message(self, message: Dict[str, Any], sender: Optional[ConversableAgent] = None) -> None:
        """Broadcast a message to all agents except the sender."""
        pass

    def _get_plan_prompt(self, team: str) -> str:
        """Get the plan prompt."""
        pass

    def _get_synthesize_prompt(self, task: str, team: str, facts: str, plan: str) -> str:
        """Get the synthesize prompt."""
        pass

    def _get_ledger_prompt(self, task: str, team: str, names: List[str]) -> str:
        """Get the ledger prompt."""
        pass

    def _get_update_facts_prompt(self, task: str, facts: str) -> str:
        """Get the update facts prompt."""
        pass

    def _get_update_plan_prompt(self, team: str) -> str:
        """Get the update plan prompt."""
        pass

    def _get_closed_book_prompt(self, task: str) -> str:
        """Get the closed book prompt."""
        pass

    def _get_team_description(self) -> str:
        """Generate a description of the team's capabilities."""
        pass

    def _get_team_names(self) -> List[str]:
        """Get the names of the team members."""
        pass

    def _initialize_task(self, task: str) -> None:
        """Initializes the task."""
        pass

    def _update_facts_and_plan(self) -> None:
        """Updates the facts and plan."""
        pass

    def update_ledger(self) -> Dict[str, Any]:
        """Updates the ledger at each turn."""
        pass

    def _prepare_final_answer(self) -> str:
        """Prepares the final answer."""
        pass

    def _select_next_agent(self, task: dict | str) -> Optional[ConversableAgent]:
        """Select the next agent to act based on the current state."""
        pass

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional["Agent"] = None,
        **kwargs: Any,
    ) -> Union[str, Dict, None]:
        """Start the orchestration process with an initial message/task."""
        return "placeholder"
