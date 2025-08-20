# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from types import TracebackType
from typing import TYPE_CHECKING, Annotated

from faststream import FastStream, Path
from faststream.nats import NatsBroker, NatsMessage

from autogen.doc_utils import export_module

if TYPE_CHECKING:
    from autogen import ConversableAgent

    from .agent import RemoteAgent


@export_module("autogen.remote")
class AgentBus:
    def __init__(self, *, agents: Iterable["ConversableAgent"] = ()):
        self.broker = NatsBroker()
        self.app = FastStream(self.broker)

        self._agents = []
        for agent in agents:
            register_agent(self.broker, agent)
            self._agents.append(agent)

    async def initiate_chat(self, chat_id: int, *remotes: "RemoteAgent") -> None:
        for remote_agent in remotes:
            await self.broker.publish(
                f"{chat_id}.{remote_agent.name}",
                {"role": "user", "content": "Hello"},
            )

    async def __aenter__(self) -> "AgentBus":
        await self.broker.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return await self.broker.__aexit__(exc_type, exc_value, traceback)

    async def run(self) -> None:
        await self.app.run()


def register_agent(broker: "NatsBroker", agent: "ConversableAgent") -> None:
    @broker.subscriber("{chat_id}." + agent.name, no_reply=True)
    async def on_message(
        message: "NatsMessage",
        chat_id: Annotated[str, Path()],
    ) -> None:
        print(await message.decode(), chat_id)
