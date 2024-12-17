# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import asyncer
import websockets

from autogen.agentchat.agent import Agent, LLMAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.function_utils import get_function_schema

from .client import Client
from .function_observer import FunctionObserver
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])


class RealtimeAgent(ConversableAgent):
    def __init__(
        self,
        name: str,
        audio_adapter: RealtimeObserver,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "ALWAYS",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        voice: str = "alloy",
    ):
        super().__init__(
            name=name,
            is_termination_msg=is_termination_msg,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            function_map=function_map,
            code_execution_config=code_execution_config,
            default_auto_reply=default_auto_reply,
            llm_config=llm_config,
            system_message=system_message,
            description=description,
            chat_messages=chat_messages,
            silent=silent,
            context_variables=context_variables,
        )
        self._client = Client(self, audio_adapter, FunctionObserver(self))
        self.llm_config = llm_config
        self.voice = voice
        self.registered_functions = {}

        self._oai_system_message = [{"content": system_message, "role": "system"}]  # todo still needed?
        self.register_reply(
            [Agent, None], RealtimeAgent.check_termination_and_human_reply, remove_other_reply_funcs=True
        )

    async def run(self):
        await self._client.run()

    def register_handover(
        self,
        *,
        description: str,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        def _decorator(func: F, name=name) -> F:
            """Decorator for registering a function to be used by an agent.

            Args:
                func: the function to be registered.

            Returns:
                The function to be registered, with the _description attribute set to the function description.

            Raises:
                ValueError: if the function description is not provided and not propagated by a previous decorator.
                RuntimeError: if the LLM config is not set up before registering a function.

            """
            # get JSON schema for the function
            name = name or func.__name__

            schema = get_function_schema(func, name=name, description=description)["function"]
            schema["type"] = "function"

            self.registered_functions[name] = (schema, func)

            return func

        return _decorator

    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        loop = asyncio.get_event_loop()
        self.answer = loop.create_future()
        loop.run_until_complete(
            self._client.send_text(
                (
                    f"I have a question for the user from the agent working on a task. DO NOT ANSWER YOURSELF, ASK THE USER AND THEN CALL 'answer_question_about_task' TO PROPAGETE THE USER ANSWER TO THE AGENT WORKING ON THE TASK. The question is: '{messages[-1]['content']}'\n\n"
                )
            )
        )

        async def get_input():
            input_text = await self.answer
            return input_text

        return True, loop.run_until_complete(get_input())
