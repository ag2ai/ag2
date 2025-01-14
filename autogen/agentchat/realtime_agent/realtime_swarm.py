# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from functools import wraps
from typing import TYPE_CHECKING, Any, Optional, Protocol

import anyio
from asyncer import create_task_group, syncify

from ..agent import Agent

if TYPE_CHECKING:
    from .clients import Role
    from .realtime_agent import RealtimeAgent

from ... import SwarmAgent

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


# todo: move to Swarm and replace ConversibleAgent typing when possible
class SwarmableProtocol(Protocol):
    def check_termination_and_human_reply(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]: ...

    # todo: whatever is needed
    def start_chat(self) -> None: ...


class SwarmableRealtimeAgent(SwarmableProtocol):
    def __init__(
        self,
        realtime_agent: RealtimeAgent,  # type: ignore
        initial_agent: SwarmAgent,
        agents: list[SwarmAgent],
    ) -> None:
        self._initial_agent = initial_agent
        self._agents = agents
        self._realtime_agent = realtime_agent

        self._answer_event: anyio.Event = anyio.Event()
        self._answer: str = ""

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
        realtime_client = self._realtime_agent._realtime_client
        await realtime_client.send_text(role=QUESTION_ROLE, text=question)

        async def _check_event_set(timeout: int = question_timeout) -> bool:
            for _ in range(timeout):
                if self._answer_event.is_set():
                    return True
                await anyio.sleep(1)
            return False

        while not await _check_event_set():
            await realtime_client.send_text(role=QUESTION_ROLE, text=question)

    def check_termination_and_human_reply(
        self,
        messages: Optional[list[dict[str, Any]]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """Check if the conversation should be terminated and if the agent should reply.

        Called when its agents turn in the chat conversation.

        Args:
            messages (list[dict[str, Any]]): The messages in the conversation.
            sender (Agent): The agent that sent the message.
            config (Optional[Any]): The configuration for the agent.
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

    def start_chat(self) -> None:
        raise NotImplementedError

    @classmethod
    def create(
        cls,
        *,
        realtime_agent: RealtimeAgent,  # type: ignore
        initial_agent: SwarmAgent,
        agents: list[SwarmAgent],
        system_message: Optional[str] = None,
    ) -> SwarmableProtocol:
        """Create a SwarmableRealtimeAgent.

        Args:
            realtime_agent (RealtimeAgent): The RealtimeAgent to create the SwarmableRealtimeAgent from.
            initial_agent (SwarmAgent): The initial agent.
            agents (list[SwarmAgent]): The agents in the swarm.
            system_message (Optional[str]): The system message to set for the agent. If None, the default system message is used.
        """
        logger = realtime_agent.logger
        if not system_message:
            if realtime_agent.system_message != "You are a helpful AI Assistant.":
                logger.warning(
                    "Overriding system message set up in `__init__`, please use `system_message` parameter of the `register_swarm` function instead."
                )
            system_message = SWARM_SYSTEM_MESSAGE

        realtime_agent._oai_system_message = [{"content": system_message, "role": "system"}]

        swarmable_agent = SwarmableRealtimeAgent(
            realtime_agent=realtime_agent, initial_agent=initial_agent, agents=agents
        )

        realtime_agent.register_realtime_function(
            name="answer_task_question", description="Answer question from the task"
        )(swarmable_agent.set_answer)

        swarmable_agent.patch_start_observers()

        return swarmable_agent

    def patch_start_observers(self) -> None:
        # todo: add middleware function and refactor the code
        org_start_observers = self._realtime_agent.start_observers

        @wraps(org_start_observers)
        async def my_start_observers(*args: Any, **kwargs: Any) -> Any:
            nonlocal org_start_observers
            # todo: do what you have to
            ...
            await org_start_observers(*args, **kwargs)

            ...

        self._realtime_agent.start_observers = my_start_observers  # type: ignore[method-assign]
