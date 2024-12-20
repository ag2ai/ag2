# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT

import base64
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

from .realtime_observer import RealtimeObserver

if TYPE_CHECKING:
    from fastapi.websockets import WebSocket

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


class TwilioAudioAdapter(RealtimeObserver):
    """Adapter for streaming audio from Twilio to OpenAI Realtime API and vice versa."""

    def __init__(self, websocket: "WebSocket"):
        """Adapter for streaming audio from Twilio to OpenAI Realtime API and vice versa.

        Args:
            websocket: WebSocket
                the websocket connection to the Twilio service
        """
        super().__init__()
        self.websocket = websocket

        # Connection specific state
        self.stream_sid = None
        self.latest_media_timestamp = 0
        self.last_assistant_item = None
        self.mark_queue: list[str] = []
        self.response_start_timestamp_twilio: Optional[int] = None

    async def update(self, response: dict[str, Any]) -> None:
        """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
        if response["type"] in LOG_EVENT_TYPES:
            logger.info(f"Received event: {response['type']}", response)

        if response.get("type") == "response.audio.delta" and "delta" in response:
            audio_payload = base64.b64encode(base64.b64decode(response["delta"])).decode("utf-8")
            audio_delta = {"event": "media", "streamSid": self.stream_sid, "media": {"payload": audio_payload}}
            await self.websocket.send_json(audio_delta)

            if self.response_start_timestamp_twilio is None:
                self.response_start_timestamp_twilio = self.latest_media_timestamp
                if SHOW_TIMING_MATH:
                    logger.info(f"Setting start timestamp for new response: {self.response_start_timestamp_twilio}ms")

            # Update last_assistant_item safely
            if response.get("item_id"):
                self.last_assistant_item = response["item_id"]

            await self.send_mark()

        # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
        if response.get("type") == "input_audio_buffer.speech_started":
            logger.info("Speech started detected.")
            if self.last_assistant_item:
                logger.info(f"Interrupting response with id: {self.last_assistant_item}")
                await self.handle_speech_started_event()

    async def handle_speech_started_event(self) -> None:
        """Handle interruption when the caller's speech starts."""
        logger.info("Handling speech started event.")
        if self.mark_queue and self.response_start_timestamp_twilio is not None:
            elapsed_time = self.latest_media_timestamp - self.response_start_timestamp_twilio
            if SHOW_TIMING_MATH:
                logger.info(
                    f"Calculating elapsed time for truncation: {self.latest_media_timestamp} - {self.response_start_timestamp_twilio} = {elapsed_time}ms"
                )

            if self.last_assistant_item:
                if SHOW_TIMING_MATH:
                    logger.info(f"Truncating item with ID: {self.last_assistant_item}, Truncated at: {elapsed_time}ms")

                truncate_event = {
                    "type": "conversation.item.truncate",
                    "item_id": self.last_assistant_item,
                    "content_index": 0,
                    "audio_end_ms": elapsed_time,
                }
                await self._client._openai_ws.send(json.dumps(truncate_event))

            await self.websocket.send_json({"event": "clear", "streamSid": self.stream_sid})

            self.mark_queue.clear()
            self.last_assistant_item = None
            self.response_start_timestamp_twilio = None

    async def send_mark(self) -> None:
        """Send a mark of audio interruption to the Twilio websocket."""
        if self.stream_sid:
            mark_event = {"event": "mark", "streamSid": self.stream_sid, "mark": {"name": "responsePart"}}
            await self.websocket.send_json(mark_event)
            self.mark_queue.append("responsePart")

    async def run(self) -> None:
        """Run the adapter.

        Start reading messages from the Twilio websocket and send audio to OpenAI.
        """
        openai_ws = self.client.openai_ws
        await self.initialize_session()

        async for message in self.websocket.iter_text():
            data = json.loads(message)
            if data["event"] == "media":
                self.latest_media_timestamp = int(data["media"]["timestamp"])
                audio_append = {"type": "input_audio_buffer.append", "audio": data["media"]["payload"]}
                await openai_ws.send(json.dumps(audio_append))
            elif data["event"] == "start":
                self.stream_sid = data["start"]["streamSid"]
                logger.info(f"Incoming stream has started {self.stream_sid}")
                self.response_start_timestamp_twilio = None
                self.latest_media_timestamp = 0
                self.last_assistant_item = None
            elif data["event"] == "mark":
                if self.mark_queue:
                    self.mark_queue.pop(0)

    async def initialize_session(self) -> None:
        """Control initial session with OpenAI."""
        session_update = {
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
        }
        await self.client.session_update(session_update)
