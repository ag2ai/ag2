# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import json

from .realtime_observer import RealtimeObserver


class FunctionObserver(RealtimeObserver):
    def __init__(self, registered_functions):
        super().__init__()
        self.registered_functions = registered_functions

    async def update(self, response):
        if response.get("type") == "response.function_call_arguments.done":
            print("!" * 50)
            print(f"Received event: {response['type']}", response)
            await self.call_function(response["call_id"], response["name"], json.loads(response["arguments"]))

    async def call_function(self, call_id, name, arguments):
        function_result = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": self.registered_functions[name](**arguments),
            },
        }
        await self.client.openai_ws.send(json.dumps(function_result))
        await self.client.openai_ws.send(json.dumps({"type": "response.create"}))

    async def run(self, openai_ws):
        await self.initialize_session(openai_ws)

    async def initialize_session(self, openai_ws):
        """Add tool to OpenAI."""
        session_update = {
            "type": "session.update",
            "session": {
                "tools": [schema for schema, _ in self.registered_functions.values()],
                "tool_choice": "auto",
            },
        }
        print("Sending session update:", json.dumps(session_update))
        await openai_ws.send(json.dumps(session_update))
