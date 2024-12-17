# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json

from asyncer import asyncify
from pydantic import BaseModel

from .realtime_observer import RealtimeObserver


class FunctionObserver(RealtimeObserver):
    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    async def update(self, response):
        if response.get("type") == "response.function_call_arguments.done":
            print("!" * 50)
            print(f"Received event: {response['type']}", response)
            await self.call_function(
                call_id=response["call_id"], name=response["name"], kwargs=json.loads(response["arguments"])
            )

    async def call_function(self, call_id, name, kwargs):
        if name in self._agent.registered_functions:
            _, func = self._agent.registered_functions[name]
            func = func if asyncio.iscoroutinefunction(func) else asyncify(func)
            try:
                result = await func(**kwargs)
            except Exception:
                result = "Function call failed"

            if isinstance(result, BaseModel):
                result = result.model_dump_json()
            elif not isinstance(result, str):
                result = json.dumps(result)

            print("!" * 50)
            print(f"Function call result: {result}")
            await self._client.function_result(call_id, result)

    async def run(self):
        await self.initialize_session()

    async def initialize_session(self):
        """Add tool to OpenAI."""
        session_update = {
            "tools": [schema for schema, _ in self._agent.registered_functions.values()],
            "tool_choice": "auto",
        }
        await self._client.session_update(session_update)
