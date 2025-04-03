# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from autogen.events.client_events import StreamEvent, UsageSummaryEvent
from autogen.messages.client_messages import (
    StreamMessage,
    UsageSummaryMessage,
)


class TestChangeUsageSummaryMessage:
    def test_deprecation(self, uuid: UUID) -> None:
        usage_summary_message = UsageSummaryMessage(uuid=uuid, actual_usage_summary=None, total_usage_summary=None)
        assert isinstance(usage_summary_message, UsageSummaryEvent)


class TestStreamMessage:
    def test_deprecation(self, uuid: UUID) -> None:
        stream_message = StreamMessage(uuid=uuid, content="random stream chunk content")
        assert isinstance(stream_message, StreamEvent)
