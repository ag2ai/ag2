# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable, Iterable, MutableMapping
from itertools import chain
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Response, status

from autogen.agentchat import ConversableAgent

from .agent_service import AgentService
from .protocol import RemoteService, RequestMessage, ResponseMessage


class HTTPAgentBus:
    def __init__(
        self,
        agents: Iterable[ConversableAgent] = (),
        *,
        long_polling_interval: float = 10.0,
        additional_services: Iterable[RemoteService] = (),
    ) -> None:
        """Create HTTPAgentBus runtime.

        Makes the passed agents capable of processing remote calls.

        Args:
            agents: Agents to register as remote services.
            long_polling_interval: Timeout to respond on task status calls for long-living executions.
                Should be less than clients' HTTP request timeout.
            additional_services: Additional services to register.
        """
        self.app = FastAPI()

        for service in chain(map(AgentService, agents), additional_services):
            register_agent_endpoints(
                app=self.app,
                service=service,
                long_polling_interval=long_polling_interval,
            )

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        """ASGI interface."""
        await self.app(scope, receive, send)


def register_agent_endpoints(
    app: FastAPI,
    service: RemoteService,
    long_polling_interval: float,
) -> None:
    tasks: dict[UUID, asyncio.Task[ResponseMessage | None]] = {}

    @app.get(f"/{service.name}" + "/{task_id}", response_model=ResponseMessage | None)
    async def remote_call_result(task_id: UUID) -> Response | ResponseMessage | None:
        if task_id not in tasks:
            raise HTTPException(
                detail=f"`{task_id}` task not found",
                status_code=status.HTTP_404_NOT_FOUND,
            )

        task = tasks[task_id]

        await asyncio.wait(
            (task, asyncio.create_task(asyncio.sleep(long_polling_interval))),
            return_when=asyncio.FIRST_COMPLETED,
        )

        if not task.done():
            return Response(status_code=status.HTTP_425_TOO_EARLY)

        try:
            reply = task.result()  # Task inner errors raising here
        finally:
            # TODO: how to clear hanged tasks?
            tasks.pop(task_id, None)

        if reply is None:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        return reply

    @app.post(f"/{service.name}", status_code=status.HTTP_202_ACCEPTED)
    async def remote_call_starter(state: RequestMessage) -> UUID:
        task, task_id = asyncio.create_task(service(state)), uuid4()
        tasks[task_id] = task
        return task_id
