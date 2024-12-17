# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

# import asyncio
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
    "You are a helpful voice assistant. Your task is to listen to user and to create tasks based on his/her inputs. E.g. if a user wishes to make change to his flight, you should create a task for it.\n"
    "DO NOT ask any additional information about the task from the user. You should assume that the user has provided all the necessary information and that the agents executing the task have all the contextual information needed.\n"
    "You have to assume that every task can be successfully completed by the swarm of agents and your only role is to create tasks for them.\n"
    "While the task is being executed, please keep the user on the line and inform him/her about the progress by calling the 'get_task_status' function. You might also get additional questions or status reports from the agents working on the task.\n"
    "Do not assume that agents cannot complete the task, unless you create a task and the agents specifically instruct you that the task is impossible to solve by them.\n"
    "Once the task is done, inform the user that the task is completed and ask if you can help with anything else.\n"
    "Do not create unethical or illegal tasks.\n"
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
        self.register_handover(name="Answer question", description="Answer question from the task")(self.set_answer)

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

        @self.register_handover(name="create_task", description="Create a task given by the user")
        async def create_task(task_input: str) -> str:
            self._client.run_task(
                asyncify(initiate_swarm_chat),
                initial_agent=initial_agent,
                agents=agents,
                user_agent=self,
                messages=task_input,
                after_work=AfterWorkOption.REVERT_TO_USER,
            )

        def _get_task_status(task_id: str) -> Generator[None, str, None]:
            while True:
                for s in [
                    "The task is in progress, agents are working on it. ETA is 1 minute",
                    "The task is successfully completed.",
                ]:
                    yield s

        it = _get_task_status("task_id")

        @self.register_handover(name="get_task_status", description="Get the status of the task")
        async def get_task_status(task_id: str) -> str:
            return next(it)

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

    def set_answer(self, answer: str) -> None:
        self._answer = answer
        self._answer_event.set()

    async def get_answer(self) -> str:
        await self._answer_event.wait()
        return self._answer

    def check_termination_and_human_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, None]]:
        print("check_termination_and_human_reply() entering")

        async def get_input():
            async with create_task_group() as tg:
                self.reset_answer()
                tg.soonify(self._client.send_text)(
                    "I have a question for the user from the agent working on a task. DO NOT ANSWER YOURSELF, "
                    "INFORM THE USER **WITH AUDIO** AND THEN CALL 'answer_question_about_task' TO PROPAGETE THE "
                    f"USER ANSWER TO THE AGENT WORKING ON THE TASK. The question is: '{messages[-1]['content']}'\n\n",
                )
                await self.get_answer()

        print("check_termination_and_human_reply() exiting")

        syncify(get_input)()

        return True, {"role": "user", "content": self._answer}

        # loop = asyncio.get_event_loop()
        # self.answer = loop.create_future()
        # # loop.run_until_complete(
        # #     self._client.send_text(
        # #         (
        # #             f"I have a question for the user from the agent working on a task. DO NOT ANSWER YOURSELF, INFORM THE USER **WITH AUDIO** AND THEN CALL 'answer_question_about_task' TO PROPAGETE THE USER ANSWER TO THE AGENT WORKING ON THE TASK. The question is: '{messages[-1]['content']}'\n\n"
        # #         )
        # #     )
        # # )
        # self.client.run_task(
        #     self._client.send_text,
        #     "I have a question for the user from the agent working on a task. DO NOT ANSWER YOURSELF, "
        #     "INFORM THE USER **WITH AUDIO** AND THEN CALL 'answer_question_about_task' TO PROPAGETE THE "
        #     f"USER ANSWER TO THE AGENT WORKING ON THE TASK. The question is: '{messages[-1]['content']}'\n\n"
        # )

        # async def get_input():
        #     input_text = await self.answer
        #     return input_text

        # return loop.run_until_complete(get_input())
