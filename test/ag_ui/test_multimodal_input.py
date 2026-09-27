# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64encode
from typing import Any

import pytest
from ag_ui.core import (
    AudioInputContent,
    DocumentInputContent,
    ImageInputContent,
    InputContentDataSource,
    InputContentUrlSource,
    TextInputContent,
    UserMessage,
    VideoInputContent,
)

from ag2 import Agent
from ag2.ag_ui import AGUIStream
from ag2.events import (
    AudioInput,
    BinaryInput,
    BinaryType,
    DocumentInput,
    ImageInput,
    ModelRequest,
    TextInput,
    UrlInput,
    VideoInput,
)
from ag2.testing import TestConfig
from test._helpers import LLMCalls

from .utils import collect_events, create_run_input

pytestmark = pytest.mark.asyncio


RAW_BYTES = b"\xff\xd8\xff\xe0"
B64_VALUE = b64encode(RAW_BYTES).decode()


def _user_request(calls: LLMCalls) -> ModelRequest:
    [messages] = calls.messages
    for m in messages:
        if isinstance(m, ModelRequest) and m.parts:
            return m
    raise AssertionError(f"No non-empty ModelRequest in {messages!r}")


async def _dispatch(*content_items: object) -> ModelRequest:
    calls = LLMCalls()
    agent = Agent("test_agent", config=TestConfig("ok"), middleware=[calls.middleware()])
    stream = AGUIStream(agent)
    run_input = create_run_input(UserMessage(id="msg_1", content=list(content_items)))

    await collect_events(stream, run_input)

    return _user_request(calls)


class TestTextContent:
    async def test_plain_string_content(self) -> None:
        calls = LLMCalls()
        agent = Agent("test_agent", config=TestConfig("ok"), middleware=[calls.middleware()])
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="hi there"))

        await collect_events(stream, run_input)

        assert _user_request(calls).parts == [TextInput("hi there")]

    async def test_text_input_content(self) -> None:
        request = await _dispatch(TextInputContent(text="hello"))

        assert request.parts == [TextInput("hello")]

    async def test_mixed_text_and_image_url(self) -> None:
        request = await _dispatch(
            TextInputContent(text="describe this"),
            ImageInputContent(source=InputContentUrlSource(value="https://x/img.png")),
        )

        assert request.parts == [
            TextInput("describe this"),
            ImageInput("https://x/img.png"),
        ]


@pytest.mark.parametrize(
    "content_cls,factory,kind,url,mime",
    [
        (ImageInputContent, ImageInput, BinaryType.IMAGE, "https://x/img.png", "image/jpeg"),
        (DocumentInputContent, DocumentInput, BinaryType.DOCUMENT, "https://x/doc.pdf", "application/pdf"),
        (AudioInputContent, AudioInput, BinaryType.AUDIO, "https://x/a.wav", "audio/wav"),
        (VideoInputContent, VideoInput, BinaryType.VIDEO, "https://x/v.mp4", "video/mp4"),
    ],
)
class TestMediaContent:
    async def test_url_source(self, content_cls: type, factory: Any, kind: BinaryType, url: str, mime: str) -> None:
        request = await _dispatch(content_cls(source=InputContentUrlSource(value=url)))

        assert request.parts == [factory(url)]
        [part] = request.parts
        assert isinstance(part, UrlInput)
        assert part.kind == kind

    async def test_data_source_base64(
        self, content_cls: type, factory: Any, kind: BinaryType, url: str, mime: str
    ) -> None:
        request = await _dispatch(content_cls(source=InputContentDataSource(value=B64_VALUE, mime_type=mime)))

        assert request.parts == [factory(data=RAW_BYTES, media_type=mime)]
        [part] = request.parts
        assert isinstance(part, BinaryInput)
        assert part.kind == kind


class TestMetadata:
    async def test_metadata_propagates_to_url_input(self) -> None:
        request = await _dispatch(
            ImageInputContent(
                source=InputContentUrlSource(value="https://x/img.png"),
                metadata={"alt": "a cat"},
            )
        )

        expected = ImageInput("https://x/img.png")
        expected.metadata = {"alt": "a cat"}
        assert request.parts == [expected]

    async def test_metadata_propagates_to_binary_input(self) -> None:
        request = await _dispatch(
            DocumentInputContent(
                source=InputContentDataSource(value=B64_VALUE, mime_type="application/pdf"),
                metadata={"source_filename": "report.pdf"},
            )
        )

        expected = DocumentInput(data=RAW_BYTES, media_type="application/pdf")
        expected.metadata = {"source_filename": "report.pdf"}
        assert request.parts == [expected]
