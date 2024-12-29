# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from openai.types.beta.realtime.realtime_server_event import RealtimeServerEvent

if TYPE_CHECKING:
    from fastapi.websockets import WebSocket

    from .realtime_agent import RealtimeAgent

from .realtime_observer import RealtimeObserver

LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]
SHOW_TIMING_MATH = False

logger = logging.getLogger(__name__)


class WebsocketAudioAdapter(RealtimeObserver):
    def __init__(self, websocket: "WebSocket"):
        super().__init__()
        self.websocket = websocket

        # Connection specific state
        self.stream_sid = None
        self.latest_media_timestamp = 0
        self.last_assistant_item: Optional[str] = None
        self.mark_queue: list[str] = []
        self.response_start_timestamp_socket: Optional[int] = None

    async def on_event(self, event: dict[str, Any]) -> None:
        """Receive events from the OpenAI Realtime API, send audio back to websocket."""
        if event["type"] in LOG_EVENT_TYPES:
            logger.info(f"Received event: {event['type']}", event)

        if event["type"] == "response.audio.delta":
            audio_payload = base64.b64encode(base64.b64decode(event["delta"])).decode("utf-8")
            audio_delta = {"event": "media", "streamSid": self.stream_sid, "media": {"payload": audio_payload}}
            await self.websocket.send_json(audio_delta)

            if self.response_start_timestamp_socket is None:
                self.response_start_timestamp_socket = self.latest_media_timestamp
                if SHOW_TIMING_MATH:
                    logger.info(f"Setting start timestamp for new response: {self.response_start_timestamp_socket}ms")

            # Update last_assistant_item safely
            if event["item_id"]:
                self.last_assistant_item = event["item_id"]

            await self.send_mark()

        # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
        if event["type"] == "input_audio_buffer.speech_started":
            logger.info("Speech started detected.")
            if self.last_assistant_item:
                logger.info(f"Interrupting response with id: {self.last_assistant_item}")
                await self.handle_speech_started_event()

    async def handle_speech_started_event(self) -> None:
        """Handle interruption when the caller's speech starts."""
        logger.info("Handling speech started event.")
        if self.mark_queue and self.response_start_timestamp_socket is not None:
            elapsed_time = self.latest_media_timestamp - self.response_start_timestamp_socket
            if SHOW_TIMING_MATH:
                logger.info(
                    f"Calculating elapsed time for truncation: {self.latest_media_timestamp} - {self.response_start_timestamp_socket} = {elapsed_time}ms"
                )

            if self.last_assistant_item:
                if SHOW_TIMING_MATH:
                    logger.info(f"Truncating item with ID: {self.last_assistant_item}, Truncated at: {elapsed_time}ms")

                await self.realtime_client.truncate_audio(
                    audio_end_ms=elapsed_time,
                    content_index=0,
                    item_id=self.last_assistant_item,
                )

            await self.websocket.send_json({"event": "clear", "streamSid": self.stream_sid})

            self.mark_queue.clear()
            self.last_assistant_item = None
            self.response_start_timestamp_socket = None

    async def send_mark(self) -> None:
        if self.stream_sid:
            mark_event = {"event": "mark", "streamSid": self.stream_sid, "mark": {"name": "responsePart"}}
            await self.websocket.send_json(mark_event)
            self.mark_queue.append("responsePart")

    async def run(self, agent: "RealtimeAgent") -> None:
        self._agent = agent
        await self.initialize_session()
        self._ready_event.set()

        async for message in self.websocket.iter_text():
            data = json.loads(message)
            if data["event"] == "media":
                self.latest_media_timestamp = int(data["media"]["timestamp"])
                await self.realtime_client.send_audio(audio=data["media"]["payload"])
            elif data["event"] == "start":
                self.stream_sid = data["start"]["streamSid"]
                logger.info(f"Incoming stream has started {self.stream_sid}")
                self.response_start_timestamp_socket = None
                self.latest_media_timestamp = 0
                self.last_assistant_item = None
            elif data["event"] == "mark":
                if self.mark_queue:
                    self.mark_queue.pop(0)

    async def initialize_session(self) -> None:
        """Control initial session with OpenAI."""
        session_update = {"input_audio_format": "pcm16", "output_audio_format": "pcm16"}  #  g711_ulaw  # "g711_ulaw",
        await self.realtime_client.session_update(session_update)
