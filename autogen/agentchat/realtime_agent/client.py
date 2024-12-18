# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

# import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

import anyio
import websockets
from asyncer import TaskGroup, asyncify, create_task_group, syncify

from autogen.agentchat.contrib.swarm_agent import AfterWorkOption, initiate_swarm_chat

from .function_observer import FunctionObserver


class Client(ABC):
    def __init__(self, agent, audio_adapter, function_observer: FunctionObserver):
        self._agent = agent
        self._observers = []
        self._openai_ws = None  # todo factor out to OpenAIClient
        self.register(audio_adapter)
        self.register(function_observer)

        # LLM config
        llm_config = self._agent.llm_config
        config = llm_config["config_list"][0]

        self.model = config["model"]
        self.temperature = llm_config["temperature"]
        self.api_key = config["api_key"]

        # create a task group to manage the tasks
        self.tg: Optional[TaskGroup] = None

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

    async def send_text(self, text: str):
        # await self._openai_ws.send(json.dumps({"type": "response.cancel"}))
        text_item = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]},
        }
        await self._openai_ws.send(json.dumps(text_item))
        await self._openai_ws.send(json.dumps({"type": "response.create"}))

    # todo override in specific clients
    async def initialize_session(self):
        """Control initial session with OpenAI."""
        session_update = {
            # todo: move to config
            "turn_detection": {"type": "server_vad"},
            "voice": self._agent.voice,
            "instructions": self._agent.system_message,
            "modalities": ["audio", "text"],
            "temperature": 0.8,
        }
        await self.session_update(session_update)

    # todo override in specific clients
    async def session_update(self, session_options):
        update = {"type": "session.update", "session": session_options}
        print("Sending session update:", json.dumps(update), flush=True)
        await self._openai_ws.send(json.dumps(update))
        print("Sending session update finished", flush=True)

    async def _read_from_client(self):
        try:
            async for openai_message in self._openai_ws:
                response = json.loads(openai_message)
                await self.notify_observers(response)
        except Exception as e:
            print(f"Error in _read_from_client: {e}")

    async def run(self):
        async with websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={self.model}",
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        ) as openai_ws:
            self._openai_ws = openai_ws
            await self.initialize_session()
            # await asyncio.gather(self._read_from_client(), *[observer.run() for observer in self._observers])
            async with create_task_group() as tg:
                self.tg = tg
                self.tg.soonify(self._read_from_client)()
                for observer in self._observers:
                    self.tg.soonify(observer.run)()
                if self._agent._start_swarm_chat:
                    self.tg.soonify(asyncify(initiate_swarm_chat))(
                        initial_agent=self._agent._initial_agent,
                        agents=self._agent._agents,
                        user_agent=self._agent,
                        messages="Find out what the user wants.",
                        after_work=AfterWorkOption.REVERT_TO_USER,
                    )

    def run_task(self, task, *args: Any, **kwargs: Any):
        self.tg.soonify(task)(*args, **kwargs)
