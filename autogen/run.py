# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import queue
import threading
from typing import Any, Callable, Iterable, Optional, Union
from uuid import UUID, uuid4

from .agentchat.agent import DEFAULT_SUMMARY_METHOD, Agent
from .agentchat.termination import MaxTurns, TerminateProtocol
from .doc_utils import export_module
from .events.agent_events import ErrorEvent, InputRequestEvent, TerminationEvent
from .events.base_event import BaseEvent
from .events.print_event import PrintEvent
from .io.base import IOStream
from .io.run_response import AsyncRunResponseProtocol, Message, RunResponseProtocol
from .run_patterns import RoundRobinRunPattern, RunPatternProtocol

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

                if isinstance(event, ErrorEvent):
                    raise event.content.error
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
    iostream: ThreadIOStream,
    response: RunResponse,
    message: str,
    chat_manager: RunPatternProtocol,
    previous_run: Optional[RunResponseProtocol],
    summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
) -> None:
    with IOStream.set_default(iostream):  # type: ignore[arg-type]
        try:
            chat_result = chat_manager.run(
                message=message,
                messages=previous_run.messages if previous_run else [],
                summary_method=summary_method,
            )

            response._summary = chat_result.summary
        except Exception as e:
            response.iostream.send(ErrorEvent(error=e))


@export_module("autogen")
def run(
    receiver: Agent,
    sender: Union[Agent, str] = "user",
    message: Optional[str] = None,
    previous_run: Optional[RunResponseProtocol] = None,
    run_pattern: Optional[RunPatternProtocol] = None,
    terminate_on: TerminateProtocol = MaxTurns(10),
    summary_method: Optional[Union[str, Callable[..., Any]]] = DEFAULT_SUMMARY_METHOD,
) -> RunResponseProtocol:
    """Run the agents with the given initial message.

    Args:
        receiver (Agent): The agent to receive the message.
        sender (Agent or str): The agent to send the message. Default is "user".
        message (str): The initial message to send to the first agent.
        previous_run (RunResponseProtocol): The previous run to continue.
        run_pattern (RunPatternProtocol): The run pattern to use.
        terminate_on (TerminateProtocol): The termination condition.
        summary_method (str or callable): a method to get a summary from the chat. Default is DEFAULT_SUMMARY_METHOD, i.e., "last_msg".
            Supported strings are "last_msg" and "reflection_with_llm":
    """
    iostream = ThreadIOStream()
    response = RunResponse(iostream)

    if run_pattern is None:
        agents = [receiver]
        if not isinstance(sender, str):
            agents = [sender] + agents

        run_pattern = RoundRobinRunPattern(*agents, terminate_on=terminate_on)

    threading.Thread(
        target=run_group_chat,
        kwargs={
            "iostream": iostream,
            "message": message,
            "chat_manager": run_pattern,
            "previous_run": previous_run,
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
