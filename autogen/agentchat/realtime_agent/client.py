# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json
from abc import ABC, abstractmethod

import websockets

from .function_observer import FunctionObserver


class Client(ABC):
    def __init__(self, agent, audio_adapter, function_observer: FunctionObserver):
        self._agent = agent
        self._observers = []
        self._openai_ws = None  # todo factor out to OpenAIClient
        self.register(audio_adapter)
        self.register(function_observer)

    def register(self, observer):
        observer.register_client(self)
        self._observers.append(observer)

    async def notify_observers(self, message):
        for observer in self._observers:
            await observer.update(message)

    async def function_result(self, call_id, result):
        result_item = {
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }
        await self._openai_ws.send(json.dumps(result_item))
        await self._openai_ws.send(json.dumps({"type": "response.create"}))

    async def send_text(self, text):
        text_item = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "system", "content": [{"type": "input_text", "text": text}]},
        }
        await self._openai_ws.send(json.dumps(text_item))
        await self._openai_ws.send(json.dumps({"type": "response.create"}))

    # todo override in specific clients
    async def initialize_session(self):
        """Control initial session with OpenAI."""
        session_update = {
            "turn_detection": {"type": "server_vad"},
            "voice": self._agent.voice,
            "instructions": self._agent.system_message,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
        await self.session_update(session_update)

    # todo override in specific clients
    async def session_update(self, session_options):
        update = {"type": "session.update", "session": session_options}
        print("Sending session update:", json.dumps(update))
        await self._openai_ws.send(json.dumps(update))

    async def _read_from_client(self):
        try:
            async for openai_message in self._openai_ws:
                response = json.loads(openai_message)
                await self.notify_observers(response)
        except Exception as e:
            print(f"Error in _read_from_client: {e}")

    async def run(self):

        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            additional_headers={
                "Authorization": f"Bearer {self._agent.llm_config['config_list'][0]['api_key']}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            self._openai_ws = openai_ws
            await self.initialize_session()
            await asyncio.gather(self._read_from_client(), *[observer.run() for observer in self._observers])
