# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json
import logging

from asyncer import asyncify
from pydantic import BaseModel

from .realtime_observer import RealtimeObserver

logger = logging.getLogger(__name__)


class FunctionObserver(RealtimeObserver):
    """Observer for handling function calls from the OpenAI Realtime API."""

    def __init__(self, agent):
        """Observer for handling function calls from the OpenAI Realtime API.

        Args:
            agent: Agent instance
                the agent to be used for the conversation
        """
        super().__init__()
        self._agent = agent

    async def update(self, response):
        """Handle function call events from the OpenAI Realtime API."""
        if response.get("type") == "response.function_call_arguments.done":
            logger.info(f"Received event: {response['type']}", response)
            await self.call_function(
                call_id=response["call_id"], name=response["name"], kwargs=json.loads(response["arguments"])
            )

    async def call_function(self, call_id, name, kwargs):
        """Call a function registered with the agent."""
        if name in self._agent.realtime_functions:
            _, func = self._agent.realtime_functions[name]
            func = func if asyncio.iscoroutinefunction(func) else asyncify(func)
            try:
                result = await func(**kwargs)
            except Exception:
                result = "Function call failed"
                logger.warning(f"Function call failed: {name}")

            if isinstance(result, BaseModel):
                result = result.model_dump_json()
            elif not isinstance(result, str):
                result = json.dumps(result)

            await self._client.function_result(call_id, result)

    async def run(self):
        """Run the observer.

        Initialize the session with the OpenAI Realtime API.
        """
        await self.initialize_session()

    async def initialize_session(self):
        """Add registered tools to OpenAI with a session update."""
        session_update = {
            "tools": [schema for schema, _ in self._agent.realtime_functions.values()],
            "tool_choice": "auto",
        }
        await self._client.session_update(session_update)
