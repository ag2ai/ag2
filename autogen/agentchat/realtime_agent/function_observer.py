# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from asyncer import asyncify
from pydantic import BaseModel

from .realtime_observer import RealtimeObserver

if TYPE_CHECKING:
    from .realtime_agent import RealtimeAgent

logger = logging.getLogger(__name__)


class FunctionObserver(RealtimeObserver):
    """Observer for handling function calls from the OpenAI Realtime API."""

    def __init__(self) -> None:
        """Observer for handling function calls from the OpenAI Realtime API.

        Args:
            agent (RealtimeAgent): The realtime agent attached to the observer.
        """
        super().__init__()

    async def on_event(self, event: dict[str, Any]) -> None:
        """Handle function call events from the OpenAI Realtime API.

        Args:
            event (dict[str, Any]): The event from the OpenAI Realtime API.
        """
        if event["type"] == "response.function_call_arguments.done":
            logger.info(f"Received event: {event['type']}", event)
            await self.call_function(
                call_id=event["call_id"],
                name=event["name"],
                kwargs=event["arguments"],
            )

    async def call_function(self, call_id: str, name: str, kwargs: dict[str, Any]) -> None:
        """Call a function registered with the agent.

        Args:
            call_id (str): The ID of the function call.
            name (str): The name of the function to call.
            kwargs (Any[str, Any]): The arguments to pass to the function.
        """

        if name in self.agent._registred_realtime_functions:
            _, func = self.agent._registred_realtime_functions[name]
            func = func if asyncio.iscoroutinefunction(func) else asyncify(func)
            try:
                result = await func(**kwargs)
            except Exception:
                result = "Function call failed"
                logger.info(f"Function call failed: {name=}, {kwargs=}", stack_info=True)

            if isinstance(result, BaseModel):
                result = result.model_dump_json()
            elif not isinstance(result, str):
                try:
                    result = json.dumps(result)
                except Exception:
                    result = str(result)

            await self.realtime_client.send_function_result(call_id, result)

    async def run(self, agent: "RealtimeAgent") -> None:
        """Run the observer.

        Initialize the session with the OpenAI Realtime API.
        """
        self._agent = agent
        await self.initialize_session()
        self._ready_event.set()

    async def initialize_session(self) -> None:
        """Add registered tools to OpenAI with a session update."""
        session_update = {
            "tools": [schema for schema, _ in self.agent._registred_realtime_functions.values()],
            "tool_choice": "auto",
        }
        await self.realtime_client.session_update(session_update)
