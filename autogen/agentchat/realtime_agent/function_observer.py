# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import json

from .realtime_agent import RealtimeObserver


class FunctionObserver(RealtimeObserver):
    def __init__(self):
        super().__init__()

    async def update(self, response):
        if response.get("type") == "response.function_call_arguments.done":
            print("!" * 50)
            print(f"Received event: {response['type']}", response)
            await self.call_function(response["call_id"], **json.loads(response["arguments"]))

    async def call_function(self, call_id, location):
        result = "The weather is cloudy." if location == "Seattle" else "The weather is sunny."
        await self.client.function_result(call_id, result)

    async def run(self):
        await self.initialize_session()

    async def initialize_session(self):
        """Add tool to OpenAI."""
        session_update = {
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                    "type": "function",
                }
            ],
            "tool_choice": "auto",
        }
        await self.client.session_update(session_update)
