# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64encode
from collections.abc import Sequence
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
from typing_extensions import Self

from ag2 import Agent, Context
from ag2.ag_ui import AGUIStream
from ag2.config import LLMClient, ModelConfig
from ag2.events import (
    AudioInput,
    BaseEvent,
    BinaryInput,
    BinaryType,
    DocumentInput,
    ImageInput,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
    UrlInput,
    VideoInput,
)

from .utils import collect_events, create_run_input

pytestmark = pytest.mark.asyncio


RAW_BYTES = b"\xff\xd8\xff\xe0"
B64_VALUE = b64encode(RAW_BYTES).decode()


class _CaptureClient(LLMClient):
    """LLM client that records the full ``messages`` list handed to the LLM.

    ``TrackingConfig`` only stores ``messages[-1]``, which in the current
    ``Agent.ask`` flow is an empty placeholder ``ModelRequest`` rather than
    the user turn we want to assert on. A custom client is needed to inspect
    the rest of the list (same pattern as ``_StreamingClient`` in
    ``test_empty_chunks.py``).
    """

    def __init__(self, captured: list[list[BaseEvent]]) -> None:
        self.captured = captured

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self.captured.append(list(messages))
        msg = ModelMessage("ok")
        await context.send(msg)
        return ModelResponse(msg)


class _CaptureConfig(ModelConfig):
    def __init__(self) -> None:
        self.captured: list[list[BaseEvent]] = []

    def copy(self) -> Self:
        return self

    def create(self) -> _CaptureClient:
        return _CaptureClient(self.captured)

    def create_files_client(self) -> None:
        raise NotImplementedError


def _user_request(config: _CaptureConfig) -> ModelRequest:
    [messages] = config.captured
    for m in messages:
        if isinstance(m, ModelRequest) and m.parts:
            return m
    raise AssertionError(f"No non-empty ModelRequest in {messages!r}")


async def _dispatch(*content_items: object) -> ModelRequest:
    config = _CaptureConfig()
    agent = Agent("test_agent", config=config)
    stream = AGUIStream(agent)
    run_input = create_run_input(UserMessage(id="msg_1", content=list(content_items)))

    await collect_events(stream, run_input)

    return _user_request(config)


class TestTextContent:
    async def test_plain_string_content(self) -> None:
        config = _CaptureConfig()
        agent = Agent("test_agent", config=config)
        stream = AGUIStream(agent)
        run_input = create_run_input(UserMessage(id="msg_1", content="hi there"))

        await collect_events(stream, run_input)

        assert _user_request(config).parts == [TextInput("hi there")]

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
