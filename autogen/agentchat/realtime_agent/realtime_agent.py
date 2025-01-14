# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from logging import Logger, getLogger
from typing import Any, Callable, Optional, TypeVar, Union

import anyio
from asyncer import asyncify, create_task_group, syncify
from fastapi import WebSocket

from ...tools import Tool
from .. import SwarmAgent
from ..agent import Agent
from ..contrib.swarm_agent import AfterWorkOption, initiate_swarm_chat
from ..conversable_agent import ConversableAgent
from .function_observer import FunctionObserver
from .oai_realtime_client import OpenAIRealtimeClient, OpenAIRealtimeWebRTCClient, Role
from .realtime_client import RealtimeClientProtocol
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])

global_logger = getLogger(__name__)

SWARM_SYSTEM_MESSAGE = (
    "You are a helpful voice assistant. Your task is to listen to user and to coordinate the tasks based on his/her inputs."
    "You can and will communicate using audio output only."
)

QUESTION_ROLE: Role = "user"
QUESTION_MESSAGE = (
    "I have a question/information for myself. DO NOT ANSWER YOURSELF, GET THE ANSWER FROM ME. "
    "repeat the question to me **WITH AUDIO OUTPUT** and AFTER YOU GET THE ANSWER FROM ME call 'answer_task_question'\n\n"
    "The question is: '{}'\n\n"
)
QUESTION_TIMEOUT_SECONDS = 20


class RealtimeAgent(ConversableAgent):
    """(Experimental) Agent for interacting with the Realtime Clients."""

    def __init__(
        self,
        *,
        name: str,
        audio_adapter: Optional[RealtimeObserver] = None,
        system_message: str = "You are a helpful AI Assistant.",
        llm_config: dict[str, Any],
        voice: str = "alloy",
        logger: Optional[Logger] = None,
        websocket: Optional[WebSocket] = None,
    ):
        """(Experimental) Agent for interacting with the Realtime Clients.

        Args:
            name (str): The name of the agent.
            audio_adapter (Optional[RealtimeObserver] = None): The audio adapter for the agent.
            system_message (str): The system message for the agent.
            llm_config (dict[str, Any], bool): The config for the agent.
            voice (str): The voice for the agent.
            websocket (Optional[WebSocket] = None): WebSocket from WebRTC javascript client
        """
        super().__init__(
            name=name,
            is_termination_msg=None,
            max_consecutive_auto_reply=None,
            human_input_mode="ALWAYS",
            function_map=None,
            code_execution_config=False,
            # no LLM config is passed down to the ConversableAgent
            llm_config=False,
            default_auto_reply="",
            description=None,
            chat_messages=None,
            silent=None,
            context_variables=None,
        )
        self._logger = logger
        self._function_observer = FunctionObserver(logger=logger)
        self._audio_adapter = audio_adapter
        self._realtime_client: RealtimeClientProtocol = OpenAIRealtimeClient(
            llm_config=llm_config, voice=voice, system_message=system_message, logger=logger
        )
        if websocket is not None:
            self._realtime_client = OpenAIRealtimeWebRTCClient(
                llm_config=llm_config, voice=voice, system_message=system_message, websocket=websocket, logger=logger
            )

        self._voice = voice

        self._observers: list[RealtimeObserver] = [self._function_observer]
        if self._audio_adapter:
            # audio adapter is not needed for WebRTC
            self._observers.append(self._audio_adapter)

        self._registred_realtime_tools: dict[str, Tool] = {}

        # is this all Swarm related?
        self._oai_system_message = [{"content": system_message, "role": "system"}]  # todo still needed? see below
        self.register_reply(
            [Agent, None], RealtimeAgent.check_termination_and_human_reply, remove_other_reply_funcs=True
        )

        self._answer_event: anyio.Event = anyio.Event()
        self._answer: str = ""
        self._start_swarm_chat = False
        self._initial_agent: Optional[SwarmAgent] = None
        self._agents: Optional[list[SwarmAgent]] = None

    def _validate_name(self, name: str) -> None:
        # RealtimeAgent does not need to validate the name
        pass

    @property
    def logger(self) -> Logger:
        """Get the logger for the agent."""
        return self._logger or global_logger

    @property
    def realtime_client(self) -> RealtimeClientProtocol:
        """Get the OpenAI Realtime Client."""
        return self._realtime_client

    @property
    def registred_realtime_tools(self) -> dict[str, Tool]:
        """Get the registered realtime tools."""
        return self._registred_realtime_tools

    def register_observer(self, observer: RealtimeObserver) -> None:
        """Register an observer with the Realtime Agent.

        Args:
            observer (RealtimeObserver): The observer to register.
        """
        self._observers.append(observer)

    def register_swarm(
        self,
        *,
        initial_agent: SwarmAgent,
        agents: list[SwarmAgent],
        system_message: Optional[str] = None,
    ) -> None:
        """Register a swarm of agents with the Realtime Agent.

        Args:
            initial_agent (SwarmAgent): The initial agent.
            agents (list[SwarmAgent]): The agents in the swarm.
            system_message (str): The system message for the agent.
        """
        logger = self.logger
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

        self.register_realtime_function(name="answer_task_question", description="Answer question from the task")(
            self.set_answer
        )

    async def run(self) -> None:
        """Run the agent."""
        # everything is run in the same task group to enable easy cancellation using self._tg.cancel_scope.cancel()
        async with create_task_group() as self._tg:
            # connect with the client first (establishes a connection and initializes a session)
            async with self._realtime_client.connect():
                # start the observers
                for observer in self._observers:
                    self._tg.soonify(observer.run)(self)

                # wait for the observers to be ready
                for observer in self._observers:
                    await observer.wait_for_ready()

                if self._start_swarm_chat and self._initial_agent and self._agents:
                    self._tg.soonify(asyncify(initiate_swarm_chat))(
                        initial_agent=self._initial_agent,
                        agents=self._agents,
                        user_agent=self,  # type: ignore[arg-type]
                        messages="Find out what the user wants.",
                        after_work=AfterWorkOption.REVERT_TO_USER,
                    )

                # iterate over the events
                async for event in self.realtime_client.read_events():
                    for observer in self._observers:
                        await observer.on_event(event)

    def register_realtime_function(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Callable[[Union[F, Tool]], Tool]:
        """Decorator for registering a function to be used by an agent.

        Args:
            name (str): The name of the function.
            description (str): The description of the function.

        Returns:
            Callable[[Union[F, Tool]], Tool]: The decorator for registering a function.
        """

        def _decorator(func_or_tool: Union[F, Tool]) -> Tool:
            """Decorator for registering a function to be used by an agent.

            Args:
                func_or_tool (Union[F, Tool]): The function or tool to register.

            Returns:
                Tool: The registered tool.
            """
            tool = Tool(func_or_tool=func_or_tool, name=name, description=description)

            self._registred_realtime_tools[tool.name] = tool

            return tool

        return _decorator

    def reset_answer(self) -> None:
        """Reset the answer event."""
        self._answer_event = anyio.Event()

    def set_answer(self, answer: str) -> str:
        """Set the answer to the question."""
        self._answer = answer
        self._answer_event.set()
        return "Answer set successfully."

    async def get_answer(self) -> str:
        """Get the answer to the question."""
        await self._answer_event.wait()
        return self._answer

    async def ask_question(self, question: str, question_timeout: int) -> None:
        """Send a question for the user to the agent and wait for the answer.
        If the answer is not received within the timeout, the question is repeated.

        Args:
            question: The question to ask the user.
            question_timeout: The time in seconds to wait for the answer.
        """
        self.reset_answer()
        await self._realtime_client.send_text(role=QUESTION_ROLE, text=question)

        async def _check_event_set(timeout: int = question_timeout) -> bool:
            for _ in range(timeout):
                if self._answer_event.is_set():
                    return True
                await anyio.sleep(1)
            return False

        while not await _check_event_set():
            await self._realtime_client.send_text(role=QUESTION_ROLE, text=question)

    def check_termination_and_human_reply(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Union[str, None]]:
        """Check if the conversation should be terminated and if the agent should reply.

        Called when its agents turn in the chat conversation.

        Args:
            messages: list of dict
                the messages in the conversation
            sender: Agent
                the agent sending the message
            config: any
                the config for the agent
        """
        if not messages:
            return False, None

        async def get_input() -> None:
            async with create_task_group() as tg:
                tg.soonify(self.ask_question)(
                    QUESTION_MESSAGE.format(messages[-1]["content"]),
                    question_timeout=QUESTION_TIMEOUT_SECONDS,
                )

        syncify(get_input)()

        return True, {"role": "user", "content": self._answer}  # type: ignore[return-value]
