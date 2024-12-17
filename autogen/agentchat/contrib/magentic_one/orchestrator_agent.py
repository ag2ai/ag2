# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Dict, List, Literal, Optional, Union

from autogen.agentchat import Agent, ConversableAgent
from autogen.logger import FileLogger

from .orchestrator_prompts import (
    ORCHESTRATOR_CLOSED_BOOK_PROMPT,
    ORCHESTRATOR_LEDGER_PROMPT,
    ORCHESTRATOR_PLAN_PROMPT,
    ORCHESTRATOR_REPLAN_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    ORCHESTRATOR_SYSTEM_MESSAGE,
    ORCHESTRATOR_UPDATE_FACTS_PROMPT,
    ORCHESTRATOR_UPDATE_PLAN_PROMPT,
)

logger: FileLogger = FileLogger(config={})


class OrchestratorAgent(ConversableAgent):
    DEFAULT_SYSTEM_MESSAGES = [{"role": "system", "content": ORCHESTRATOR_SYSTEM_MESSAGE}]

    def __init__(
        self,
        name: str,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = {
            "work_dir": "coding",
            "use_docker": False,
        },
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
        replan_prompt: str = ORCHESTRATOR_REPLAN_PROMPT,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        agents: Optional[List[ConversableAgent]] = [],
        max_rounds: int = 20,
        max_stalls_before_replan: int = 3,
        max_replans: int = 3,
        return_final_answer: bool = False,
        agent_whole_history: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name,
            system_message=system_messages[0]["content"],
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=False,
            llm_config=llm_config,
            default_auto_reply=default_auto_reply,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
        )
        pass

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        **kwargs: Any,
    ) -> Union[str, Dict, None]:
        """Start the orchestration process with an initial message/task."""
        raise NotImplementedError
