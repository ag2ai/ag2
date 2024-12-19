# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

# import asyncio
import json
import logging
from typing import Any, Optional

import anyio
import websockets
from asyncer import TaskGroup, asyncify, create_task_group, syncify

from autogen.agentchat.contrib.swarm_agent import AfterWorkOption, initiate_swarm_chat

from .function_observer import FunctionObserver

logger = logging.getLogger(__name__)


class OpenAIRealtimeClient:
    """(Experimental) Client for OpenAI Realtime API."""

    def __init__(self, agent, audio_adapter, function_observer: FunctionObserver):
        """(Experimental) Client for OpenAI Realtime API.

        args:
            agent: Agent instance
                the agent to be used for the conversation
            audio_adapter: RealtimeObserver
                adapter for streaming the audio from the client
            function_observer: FunctionObserver
                observer for handling function calls
        """
        self._agent = agent
        self._observers = []
        self._openai_ws = None  # todo factor out to OpenAIClient
        self.register(audio_adapter)
        self.register(function_observer)

        # LLM config
        llm_config = self._agent.llm_config

        print("!" * 100)
        print(llm_config)
        config = llm_config["config_list"][0]

        self.model = config["model"]
        self.temperature = llm_config["temperature"]
        self.api_key = config["api_key"]

        # create a task group to manage the tasks
        self.tg: Optional[TaskGroup] = None

    def register(self, observer):
        """Register an observer to the client."""
        observer.register_client(self)
        self._observers.append(observer)

    async def notify_observers(self, message):
        """Notify all observers of a message from the OpenAI Realtime API."""
        for observer in self._observers:
            await observer.update(message)

    async def function_result(self, call_id, result):
        """Send the result of a function call to the OpenAI Realtime API."""
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

    async def send_text(self, *, role: str, text: str):
        """Send a text message to the OpenAI Realtime API."""
        await self._openai_ws.send(json.dumps({"type": "response.cancel"}))
        text_item = {
            "type": "conversation.item.create",
            "item": {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]},
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
        """Send a session update to the OpenAI Realtime API."""
        update = {"type": "session.update", "session": session_options}
        logger.info("Sending session update:", json.dumps(update))
        await self._openai_ws.send(json.dumps(update))
        logger.info("Sending session update finished")

    async def _read_from_client(self):
        """Read messages from the OpenAI Realtime API."""
        try:
            async for openai_message in self._openai_ws:
                response = json.loads(openai_message)
                await self.notify_observers(response)
        except Exception as e:
            logger.warning(f"Error in _read_from_client: {e}")

    async def run(self):
        """Run the client."""
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
