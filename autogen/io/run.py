# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import multiprocessing
import queue
from typing import Any, Iterable, Optional
from uuid import UUID, uuid4

from ..agentchat import Agent, GroupChat, GroupChatManager
from ..messages.print_message import PrintMessage
from ..messages.run_events import Event, InputRequestEvent, Message, TerminationEvent, get_event
from .base import IOStream
from .run_response import AsyncRunResponseProtocol, RunResponseProtocol


class MultiprocessingIOStream:
    def __init__(self) -> None:
        self._input_stream: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]
        self._output_stream: multiprocessing.Queue = multiprocessing.Queue()  # type: ignore[type-arg]

    def input(self, prompt: str = "", *, password: bool = False) -> str:
        # if password:
        #     return getpass.getpass(prompt if prompt != "" else "Password: ")
        self.send(InputRequestEvent(uuid=uuid4(), prompt=prompt))
        return self._output_stream.get()  # type: ignore[no-any-return]

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        print_message = PrintMessage(*objects, sep=sep, end=end)
        self.send(print_message)

    def send(self, message: Any) -> None:
        self._input_stream.put(message.model_dump())

    @property
    def input_stream(self) -> multiprocessing.Queue:  # type: ignore[type-arg]
        return self._input_stream


class RunResponse:
    def __init__(self, iostream: MultiprocessingIOStream):
        self.iostream = iostream
        self._summary: Optional[str] = None
        self._uuid = uuid4()

    def _queue_generator(self, q: multiprocessing.Queue) -> Iterable[Event]:  # type: ignore[type-arg]
        """A generator to yield items from the queue until the termination message is found."""
        while True:
            try:
                # Get an item from the queue
                item = q.get(timeout=0.1)  # Adjust timeout as needed
                event = get_event(item)

                if isinstance(event, TerminationEvent):
                    self._summary = event.summary
                    break

                if isinstance(event, InputRequestEvent):
                    event.respond = lambda response: self.iostream._output_stream.put(response)

                yield event
            except queue.Empty:
                continue  # Wait for more items in the queue

    @property
    def events(self) -> Iterable[Event]:
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


def run_single_agent(agent: Agent, iostream: MultiprocessingIOStream, message: str, **kwargs: Any) -> None:
    with IOStream.set_default(iostream):  # type: ignore[arg-type]
        chat_result = agent.run(message=message, **kwargs)  # type: ignore[attr-defined]
        iostream.send(TerminationEvent(uuid=uuid4(), summary=chat_result.summary))


def run_group_chat(*agents: Agent, iostream: MultiprocessingIOStream, message: str, **kwargs: Any) -> None:
    with IOStream.set_default(iostream):  # type: ignore[arg-type]
        groupchat = GroupChat(agents=agents, speaker_selection_method="auto", messages=[])

        manager = GroupChatManager(
            name="group_manager",
            groupchat=groupchat,
            llm_config=agents[0].llm_config,
            is_termination_msg=lambda x: "DONE!" in (x.get("content", "") or "").upper(),
        )

        chat_result = agents[0].initiate_chat(
            recipient=manager,
            message=message,
        )

        iostream.send(TerminationEvent(uuid=uuid4(), summary=chat_result.summary))


def run(
    *agents: Agent,
    initial_message: Optional[str] = None,
    previous_run: Optional[RunResponseProtocol] = None,
    **kwargs: Any,
) -> RunResponseProtocol:
    """Run the agents with the given initial message.

    Args:
        agents (Agent): The agents to run.
        initial_message (str): The initial message to send to the first agent.
        previous_run (RunResponseProtocol): The previous run to continue.
        kwargs: Additional arguments to pass to the agents.

    """
    iostream = MultiprocessingIOStream()
    response = RunResponse(iostream)

    if len(agents) == 1:
        process = multiprocessing.Process(
            target=run_single_agent, args=(agents[0], iostream, initial_message), kwargs=kwargs
        )
        process.start()

    else:
        process = multiprocessing.Process(
            target=run_group_chat, args=(agents), kwargs={**kwargs, "iostream": iostream, "message": initial_message}
        )
        process.start()

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
