# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import getpass
from typing import TYPE_CHECKING

from autogen.io.run_response import RunResponseProtocol

from ...doc_utils import export_module
from ...events.agent_events import InputRequestEvent
from ...events.base_event import BaseEvent
from .event_processor import EventProcessorProtocol


@export_module("autogen.io.event_processors")
class ConsoleEventProcessor:
    def process(self, response: RunResponseProtocol) -> None:
        for event in response.events:
            self.process_event(event)

    def process_event(self, event: BaseEvent) -> None:
        if isinstance(event, InputRequestEvent):
            prompt = event.content.prompt  # type: ignore[attr-defined]
            if event.content.password:  # type: ignore[attr-defined]
                result = getpass.getpass(prompt if prompt != "" else "Password: ")
            result = input(prompt)
            event.content.respond(result)  # type: ignore[attr-defined]
        else:
            event.print()


if TYPE_CHECKING:

    def check_group_chat_manager_implements_chat_manager_protocol(x: ConsoleEventProcessor) -> EventProcessorProtocol:
        return x
