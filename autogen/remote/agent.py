# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from uuid import uuid4

import httpx

from autogen import Agent

from .protocol import NextSpeakerEvent, ProtocolEvents, SendEvent, StopEvent, serialize_event


class RemoteAgent(Agent):
    def __init__(self, url: str, name: str | None = None) -> None:
        self.url = url
        self.headers = {}

        self._name = name or uuid4().hex

        self.client_cache = None
        self.silent = True

    @property
    def name(self) -> str:
        return self._name

    def _raise_exception_on_async_reply_functions(self) -> None:
        pass

    def _prepare_chat(
        self,
        recipient: "Agent",
        chat_id: int,
        clear_history: bool = False,
        prepare_recipient: bool = True,
        reply_at_receive: bool = True,
    ) -> None:
        # TODO: set headers in `receive` after explicit `chat_id` option added
        self.headers["X-Chat-Id"] = str(chat_id)

    def receive(
        self,
        message: dict[str, Any] | str,
        sender: "Agent",
        request_reply: bool | None = None,
        silent: bool | None = False,
    ) -> None:
        # update RemoteAgent state
        httpx.post(
            self.url,
            content=SendEvent(content=message).model_dump_json(),
            headers=self.headers,
            timeout=30,
        )

        # command RemoteAgent to talk
        reply_response = httpx.post(
            self.url,
            content=NextSpeakerEvent().model_dump_json(),
            headers=self.headers,
            timeout=30,
        )

        event = serialize_event(reply_response.json())

        if event.event_type is ProtocolEvents.STOP_CHAT:
            # confirm chat stopped
            httpx.post(
                self.url,
                content=StopEvent().model_dump_json(),
                headers=self.headers,
                timeout=30,
            )
            self.headers.clear()

        if event.event_type is ProtocolEvents.SEND_MESSAGE:
            sender.receive(event.content, self)
