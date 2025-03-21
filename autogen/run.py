# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import queue
import threading
from typing import Any, Callable, Iterable, Optional, Union
from uuid import UUID, uuid4

from .agentchat.agent import DEFAULT_SUMMARY_METHOD, Agent
from .agentchat.conversable_agent import ConversableAgent
from .chat_managers import ChatManagerProtocol, RoundRobinChatManager
from .doc_utils import export_module
from .events.agent_events import InputRequestEvent, TerminationEvent
from .events.base_event import BaseEvent
from .events.print_event import PrintEvent
from .io.base import IOStream
from .io.run_response import AsyncRunResponseProtocol, Message, RunResponseProtocol

__all__ = ["run"]


class ThreadIOStream:
    def __init__(self) -> None:
        self._input_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]
        self._output_stream: queue.Queue = queue.Queue()  # type: ignore[type-arg]

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        # if password:
        #     return getpass.getpass(prompt if prompt != "" else "Password: ")
        self.send(InputRequestEvent(prompt=prompt, password=password))
        return self._output_stream.get()  # type: ignore[no-any-return]

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print_message = PrintEvent(*objects, sep=sep, end=end)
        self.send(print_message)

    def send(self, message: Any) -> None:
        self._input_stream.put(message)

    @property
    def input_stream(self) -> queue.Queue:  # type: ignore[type-arg]
        return self._input_stream


class RunResponse:
    def __init__(self, iostream: ThreadIOStream):
        self.iostream = iostream
        self._summary: Optional[str] = None
        self._uuid = uuid4()

    def _queue_generator(self, q: queue.Queue) -> Iterable[BaseEvent]:  # type: ignore[type-arg]
        """A generator to yield items from the queue until the termination message is found."""
        while True:
            try:
                # Get an item from the queue
                event = q.get(timeout=0.1)  # Adjust timeout as needed
                # event = get_event(item)

                if isinstance(event, InputRequestEvent):
                    event.content.respond = lambda response: self.iostream._output_stream.put(response)

                yield event

                if isinstance(event, TerminationEvent):
                    break
            except queue.Empty:
                continue  # Wait for more items in the queue

    @property
    def events(self) -> Iterable[BaseEvent]:
        return self._queue_generator(self.iostream.input_stream)

    @property
    def messages(self) -> Iterable[Message]:
        return []

    @property
    def summary(self) -> Optional[str]:
        return self._summary

    @property
    def above_run(self) -> Optional["RunResponseProtocol"]:
        return None

    @property
    def uuid(self) -> UUID:
        return self._uuid


def run_group_chat(
    *agents: Agent,
    iostream: ThreadIOStream,
    response: RunResponse,
    message: str,
    chat_manager: ChatManagerProtocol,
    previous_run: Optional[RunResponseProtocol],
    max_turns: Optional[int] = None,
    summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
) -> None:
    with IOStream.set_default(iostream):  # type: ignore[arg-type]
        chat_result = chat_manager.run(
            *agents,
            message=message,
            messages=previous_run.messages if previous_run else [],
            max_turns=max_turns if max_turns else 10,
            summary_method=summary_method,
        )

        response._summary = chat_result.summary


@export_module("autogen")
def run(
    *agents: Agent,
    initial_message: Optional[str] = None,
    previous_run: Optional[RunResponseProtocol] = None,
    chat_manager: Optional[ChatManagerProtocol] = None,
    max_turns: Optional[int] = None,
    summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
    user_input: bool = False,
) -> RunResponseProtocol:
    """Run the agents with the given initial message.

    Args:
        agents (Agent): The agents to run.
        initial_message (str): The initial message to send to the first agent.
        previous_run (RunResponseProtocol): The previous run to continue.
        chat_manager (ChatManagerProtocol): The chat manager to use for the group chat.
        clear_history (bool): Whether to clear the history of the agents.
        max_turns (int): The maximum number of turns to run.
        summary_method (Union[str, Callable[..., Any]]): The method to use to summarize the chat.
        summary_args (dict[str, Any]): The arguments to pass to the summary method.
        cache (AbstractCache): The cache to use.
        tools (Union[Tool, Iterable[Tool]]): The tools to use.
        executor_kwargs (dict[str, Any]): The arguments to pass to the executor.
        user_input (bool): Whether to allow user input.
        kwargs: Additional arguments to pass to the agents.
    """
    iostream = ThreadIOStream()
    response = RunResponse(iostream)

    if user_input:
        agents = (ConversableAgent(name="user", human_input_mode="ALWAYS"),) + agents

    for agent in agents:
        for tool in agent.tools:
            tool.register_for_execution(agent)

    threading.Thread(
        target=run_group_chat,
        args=agents,
        kwargs={
            "iostream": iostream,
            "message": initial_message,
            "chat_manager": chat_manager if chat_manager else RoundRobinChatManager(),
            "previous_run": previous_run,
            "max_turns": max_turns,
            "summary_method": summary_method,
            "response": response,
        },
    ).start()

    return response


async def a_run(  # type: ignore[empty-body]
    *agents: Agent,
    initial_message: Optional[str] = None,
    previous_run: Optional[RunResponseProtocol] = None,
    **kwargs: Any,
) -> AsyncRunResponseProtocol:
    """Run the agents with the given initial message.

    Args:
        agents (Agent): The agents to run.
        initial_message (str): The initial message to send to the first agent.
        previous_run (RunResponseProtocol): The previous run to continue.
        kwargs: Additional arguments to pass to the agents

    """
    ...
