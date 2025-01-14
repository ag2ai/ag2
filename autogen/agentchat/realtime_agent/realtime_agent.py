# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from logging import Logger, getLogger
from typing import Any, Callable, Optional, TypeVar, Union

from asyncer import create_task_group

from ...tools import Tool
from ..agent import Agent
from ..conversable_agent import ConversableAgent
from .clients.realtime_client import RealtimeClientProtocol, get_client
from .function_observer import FunctionObserver
from .realtime_observer import RealtimeObserver

F = TypeVar("F", bound=Callable[..., Any])

global_logger = getLogger(__name__)


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
        **client_kwargs: Any,
    ):
        """(Experimental) Agent for interacting with the Realtime Clients.

        Args:
            name (str): The name of the agent.
            audio_adapter (Optional[RealtimeObserver] = None): The audio adapter for the agent.
            system_message (str): The system message for the agent.
            llm_config (dict[str, Any], bool): The config for the agent.
            voice (str): The voice for the agent.
            logger (Optional[Logger]): The logger for the agent.
            **client_kwargs (Any): The keyword arguments for the client.
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

        self._realtime_client: RealtimeClientProtocol = get_client(
            llm_config=llm_config, voice=voice, system_message=system_message, logger=self.logger, **client_kwargs
        )

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

    async def start_observers(self) -> None:
        for observer in self._observers:
            self._tg.soonify(observer.run)(self)

        # wait for the observers to be ready
        for observer in self._observers:
            await observer.wait_for_ready()

    async def run(self) -> None:
        """Run the agent."""
        # everything is run in the same task group to enable easy cancellation using self._tg.cancel_scope.cancel()
        async with create_task_group() as self._tg:
            # connect with the client first (establishes a connection and initializes a session)
            async with self._realtime_client.connect():
                # start the observers and wait for them to be ready
                await self.start_observers()

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
