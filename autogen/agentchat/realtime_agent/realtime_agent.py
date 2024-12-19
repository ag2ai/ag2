# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

# import asyncio
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, TypeVar, Union

import anyio
import websockets
from asyncer import TaskGroup, asyncify, create_task_group, syncify

from autogen import ON_CONDITION, AfterWorkOption, SwarmAgent, initiate_swarm_chat
from autogen.agentchat.agent import Agent, LLMAgent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.function_utils import get_function_schema

from .client import Client
from .function_observer import FunctionObserver
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

SWARM_SYSTEM_MESSAGE = (
    "You are a helpful voice assistant. Your task is to listen to user and to coordinate the tasks based on his/her inputs."
    "Only call the 'answer_task_question' function when you have the answer from the user."
    "You can communicate and will communicate using audio output only."
)

QUESTION_ROLE = "user"
QUESTION_MESSAGE = (
    "I have a question/information for myself. DO NOT ANSWER YOURSELF, GET THE ANSWER FROM ME. "
    "repeat the question to me **WITH AUDIO OUTPUT** and then call 'answer_task_question' AFTER YOU GET THE ANSWER FROM ME\n\n"
    "The question is: '{}'\n\n"
)


class RealtimeAgent(ConversableAgent):
    def __init__(
        self,
        *,
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

        self._answer_event: anyio.Event = anyio.Event()
        self._answer: str = ""
        self._start_swarm_chat = False
        self._initial_agent = None
        self._agents = None

    def register_swarm(
        self,
        *,
        initial_agent: SwarmAgent,
        agents: List[SwarmAgent],
        system_message: Optional[str] = None,
    ) -> None:
        if not system_message:
            if self.system_message != "You are a helpful AI Assistant.":
                logger.warning(
                    "Overriding system message set up in `__init__`, please use `system_message` parameter of the `register_swarm` function instead."
                )
            system_message = SWARM_SYSTEM_MESSAGE

        self._oai_system_message = [{"content": system_message, "role": "system"}]

        self._start_swarm_chat = True
        self._initial_agent = initial_agent
        self._agents = agents

        # def _get_task_status(task_id: str) -> Generator[None, str, None]:
        #     while True:
        #         for s in [
        #             "The task is in progress, agents are working on it. ETA is 1 minute",
        #             "The task is successfully completed.",
        #         ]:
        #             yield s

        # it = _get_task_status("task_id")

        # @self.register_handover(name="get_task_status", description="Get the status of the task")
        # async def get_task_status(task_id: str) -> str:
        #     return next(it)

        self.register_handover(name="answer_task_question", description="Answer question from the task")(
            self.set_answer
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

    def reset_answer(self) -> None:
        self._answer_event = anyio.Event()

    def set_answer(self, answer: str) -> str:
        self._answer = answer
        self._answer_event.set()
        return "Answer set successfully."

    async def get_answer(self) -> str:
        await self._answer_event.wait()
        return self._answer

    async def ask_question(self, question: str, question_timeout: int) -> str:
        self.reset_answer()
        await anyio.sleep(1)
        await self._client.send_text(role=QUESTION_ROLE, text=question)

        async def _check_event_set(timeout: int = question_timeout) -> None:
            for _ in range(timeout):
                if self._answer_event.is_set():
                    return True
                await anyio.sleep(1)
            return False

        while not await _check_event_set():
            await self._client.send_text(role=QUESTION_ROLE, text=question)

    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        async def get_input():
            async with create_task_group() as tg:
                tg.soonify(self.ask_question)(QUESTION_MESSAGE.format(messages[-1]["content"]), 20)

        syncify(get_input)()

        return True, {"role": "user", "content": self._answer}
