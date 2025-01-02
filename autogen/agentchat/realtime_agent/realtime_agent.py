# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from logging import Logger, getLogger
from typing import Any, Callable, Literal, Optional, TypeVar, Union

import anyio
from asyncer import create_task_group, syncify

from autogen import SwarmAgent
from autogen.agentchat.agent import Agent
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.function_utils import get_function_schema

from .function_observer import FunctionObserver
from .oai_realtime_client import OpenAIRealtimeClient, Role
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])

global_logger = getLogger(__name__)

SWARM_SYSTEM_MESSAGE = (
    "You are a helpful voice assistant. Your task is to listen to user and to coordinate the tasks based on his/her inputs."
    "Only call the 'answer_task_question' function when you have the answer from the user."
    "You can communicate and will communicate using audio output only."
)

QUESTION_ROLE: Role = "user"
QUESTION_MESSAGE = (
    "I have a question/information for myself. DO NOT ANSWER YOURSELF, GET THE ANSWER FROM ME. "
    "repeat the question to me **WITH AUDIO OUTPUT** and then call 'answer_task_question' AFTER YOU GET THE ANSWER FROM ME\n\n"
    "The question is: '{}'\n\n"
)
QUESTION_TIMEOUT_SECONDS = 20


class RealtimeAgent(ConversableAgent):
    """(Experimental) Agent for interacting with the Realtime Clients."""

    def __init__(
        self,
        *,
        name: str,
        audio_adapter: RealtimeObserver,
        system_message: str = "You are a helpful AI Assistant.",
        llm_config: dict[str, Any],
        voice: str = "alloy",
        logger: Optional[Logger] = None,
    ):
        """(Experimental) Agent for interacting with the Realtime Clients.

        Args:
            name (str): The name of the agent.
            audio_adapter (RealtimeObserver): The audio adapter for the agent.
            system_message (str): The system message for the agent.
            llm_config (dict[str, Any], bool): The config for the agent.
            voice (str): The voice for the agent.
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
        self._realtime_client = OpenAIRealtimeClient(
            llm_config=llm_config, voice=voice, system_message=system_message, logger=logger
        )
        self._voice = voice

        self._observers: list[RealtimeObserver] = [self._function_observer, self._audio_adapter]

        self._registred_realtime_functions: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

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

    @property
    def logger(self) -> Logger:
        """Get the logger for the agent."""
        return self._logger or global_logger

    @property
    def realtime_client(self) -> OpenAIRealtimeClient:
        """Get the OpenAI Realtime Client."""
        return self._realtime_client

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

                # iterate over the events
                async for event in self.realtime_client.read_events():
                    for observer in self._observers:
                        await observer.on_event(event)

    def register_realtime_function(
        self,
        *,
        description: str,
        name: Optional[str] = None,
    ) -> Callable[[F], F]:
        def _decorator(func: F, name: Optional[str] = name) -> F:
            """Decorator for registering a function to be used by an agent.

            Args:
                func (callable[..., Any]): the function to be registered.
                name (str): the name of the function.

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

            self._registred_realtime_functions[name] = (schema, func)

            return func

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
        """
        Send a question for the user to the agent and wait for the answer.
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
