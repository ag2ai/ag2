# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import websockets

from autogen.agentchat.agent import Agent, LLMAgent


class RealtimeObserver(ABC):
    def __init__(self):
        self.client = None

    def register_client(self, client):
        self.client = client

    @abstractmethod
    async def run(self, openai_ws):
        pass

    @abstractmethod
    async def update(self, message):
        pass


class RealtimeAgent(LLMAgent):
    def __init__(
        self,
        name: str,
        audio_adapter: RealtimeObserver,
        system_message: Optional[Union[str, List]] = "You are a helpful AI Assistant.",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "TERMINATE",
        function_map: Optional[Dict[str, Callable]] = None,
        code_execution_config: Union[Dict, Literal[False]] = False,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        default_auto_reply: Union[str, Dict] = "",
        description: Optional[str] = None,
        chat_messages: Optional[Dict[Agent, List[Dict]]] = None,
        silent: Optional[bool] = None,
        context_variables: Optional[Dict[str, Any]] = None,
        voice: str = "alloy",
    ):
        self.llm_config = llm_config
        self.system_message = system_message
        self.voice = voice
        self.observers = []
        self.openai_ws = None

        self.register(audio_adapter)

    def register(self, observer):
        observer.register_client(self)
        self.observers.append(observer)

    async def notify_observers(self, message):
        for observer in self.observers:
            await observer.update(message)

    async def _read_from_client(self, openai_ws):
        try:
            async for openai_message in openai_ws:
                response = json.loads(openai_message)
                await self.notify_observers(response)
        except Exception as e:
            print(f"Error in _read_from_client: {e}")

    async def run(self):
        async with websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01",
            extra_headers={"Authorization": f"Bearer {self.llm_config[0]['api_key']}", "OpenAI-Beta": "realtime=v1"},
        ) as openai_ws:
            self.openai_ws = openai_ws
            self.initialize_session()
            await asyncio.gather(
                self._read_from_client(openai_ws), *[observer.run(openai_ws) for observer in self.observers]
            )

    async def initialize_session(self):
        """Control initial session with OpenAI."""
        session_update = {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "server_vad"},
                "input_audio_format": "g711_ulaw",
                "output_audio_format": "g711_ulaw",
                "voice": self.voice,
                "instructions": self.system_message,
                "modalities": ["text", "audio"],
                "temperature": 0.8,
            },
        }
        print("Sending session update:", json.dumps(session_update))
        await self.openai_ws.send(json.dumps(session_update))
