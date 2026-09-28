# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A server tool's result as the client receives it: in content parts, or as text before 1.0."""

import logging
from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import PROTOCOL_VERSION, UserMessage

from ag2 import Agent, ToolResult
from ag2.ag_ui import AGUIStream
from ag2.events import (
    AudioInput,
    BinaryInput,
    DataInput,
    DocumentInput,
    FileIdInput,
    ImageInput,
    Input,
    TextInput,
    ToolCallEvent,
    UrlInput,
)
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_run, only, run_input

pytestmark = pytest.mark.asyncio

PNG = b"\x89PNG\r\n"
PNG_B64 = b64encode(PNG).decode()


async def _result_content(
    result: ToolResult, *, protocol_version: str | None = PROTOCOL_VERSION
) -> str | list[dict[str, Any]]:
    """What the client is sent for a tool that returns `result`."""
    agent = Agent("test_agent", config=TestConfig(ToolCallEvent(name="produce"), "done"))

    @agent.tool
    def produce() -> ToolResult:
        return result

    events = await dispatch_run(
        AGUIStream(agent), run_input(UserMessage(id="m1", content="go"), protocol_version=protocol_version)
    )
    return only(events, "TOOL_CALL_RESULT")["content"]


class TestAOneZeroClient:
    async def test_a_lone_text_is_a_plain_string(self) -> None:
        assert await _result_content(ToolResult("sunny")) == "sunny"

    async def test_text_beside_an_image_arrives_as_both_parts(self) -> None:
        content = await _result_content(ToolResult("the chart", ImageInput(data=PNG, media_type="image/png")))

        assert content == [
            {"type": "text", "text": "the chart"},
            {"type": "image", "source": {"type": "data", "value": PNG_B64, "mimeType": "image/png"}},
        ]

    async def test_structured_output_travels_as_its_text(self) -> None:
        """The protocol has no JSON part."""
        content = await _result_content(ToolResult(DataInput({"temp": 22}), TextInput("ok")))

        assert content == [{"type": "text", "text": '{"temp":22}'}, {"type": "text", "text": "ok"}]

    @pytest.mark.parametrize(
        "part,expected",
        [
            (
                ImageInput("https://x/a.png"),
                {"type": "image", "source": {"type": "url", "value": "https://x/a.png"}},
            ),
            (
                AudioInput(data=b"RIFF", media_type="audio/wav"),
                {
                    "type": "audio",
                    "source": {"type": "data", "value": b64encode(b"RIFF").decode(), "mimeType": "audio/wav"},
                },
            ),
            (
                DocumentInput("https://x/a.pdf"),
                {"type": "document", "source": {"type": "url", "value": "https://x/a.pdf"}},
            ),
            (
                UrlInput("https://x/blob", kind="binary"),
                {"type": "document", "source": {"type": "url", "value": "https://x/blob"}},
            ),
            (
                BinaryInput(b"\x00", media_type="application/octet-stream"),
                {
                    "type": "document",
                    "source": {
                        "type": "data",
                        "value": b64encode(b"\x00").decode(),
                        "mimeType": "application/octet-stream",
                    },
                },
            ),
            (
                FileIdInput("file-abc"),
                {"type": "document", "source": {"type": "file", "value": "file-abc"}},
            ),
        ],
    )
    async def test_each_part_becomes_the_content_part_of_its_kind(self, part: Input, expected: dict[str, Any]) -> None:
        assert await _result_content(ToolResult("see", part)) == [{"type": "text", "text": "see"}, expected]

    async def test_metadata_carries_over(self) -> None:
        image = ImageInput("https://x/a.png")
        image.metadata = {"alt": "a chart"}

        content = await _result_content(ToolResult(image))

        assert content == [
            {"type": "image", "source": {"type": "url", "value": "https://x/a.png"}, "metadata": {"alt": "a chart"}}
        ]

    async def test_a_lone_text_with_metadata_keeps_it_in_a_part(self) -> None:
        """A plain string has nowhere to put metadata, and dropping it would lose it."""
        text = TextInput("sunny")
        text.metadata = {"source": "met office"}

        content = await _result_content(ToolResult(text))

        assert content == [{"type": "text", "text": "sunny", "metadata": {"source": "met office"}}]

    async def test_a_result_of_nothing_is_the_empty_string(self) -> None:
        assert await _result_content(ToolResult()) == ""


class TestAClientPredatingOneZero:
    """Its schema reads a tool result as a string, so it gets the text and nothing invented."""

    async def test_a_lone_text_is_the_same_string(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            content = await _result_content(ToolResult("sunny"), protocol_version=None)

        assert content == "sunny"
        assert caplog.records == []

    async def test_media_are_dropped_and_the_loss_is_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        result = ToolResult(
            "the chart",
            ImageInput(data=PNG, media_type="image/png"),
            DataInput({"temp": 22}),
            FileIdInput("file-abc"),
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            content = await _result_content(result, protocol_version=None)

        assert content == 'the chart\n{"temp":22}'
        [warning] = caplog.records
        assert "document" in warning.getMessage()
        assert "image" in warning.getMessage()
        assert "@ag-ui/* 1.0" in warning.getMessage()

    async def test_a_result_of_media_only_is_the_empty_string(self) -> None:
        content = await _result_content(ToolResult(ImageInput(data=PNG, media_type="image/png")), protocol_version=None)

        assert content == ""
