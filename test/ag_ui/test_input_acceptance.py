# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Keep AG-UI's filtering aligned with the provider mappers it protects."""

from collections.abc import Callable

import pytest
from fast_depends.use import SerializerCls

from ag2 import ToolResult
from ag2.ag_ui.input_acceptance import accepts_input, input_modalities
from ag2.config import (
    AnthropicConfig,
    BedrockConfig,
    DashScopeConfig,
    GeminiConfig,
    MistralConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenAIResponsesConfig,
    TypeSafeConfig,
    VertexAIConfig,
    XAIConfig,
    ZAIConfig,
)
from ag2.config.anthropic.mappers import convert_messages as anthropic_messages
from ag2.config.bedrock.mappers import convert_messages as bedrock_messages
from ag2.config.dashscope.mappers import convert_messages as dashscope_messages
from ag2.config.gemini.mappers import convert_messages as gemini_messages
from ag2.config.mistral.mappers import convert_messages as mistral_messages
from ag2.config.ollama.mappers import convert_messages as ollama_messages
from ag2.config.openai.mappers import convert_messages as openai_messages
from ag2.config.openai.mappers import events_to_responses_input as responses_messages
from ag2.config.typesafe.mappers import _parts_content as typesafe_parts
from ag2.config.xai.mappers import convert_messages as xai_messages
from ag2.config.zai.mappers import convert_messages as zai_messages
from ag2.events import (
    BinaryInput,
    BinaryType,
    FileIdInput,
    ModelRequest,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
)
from ag2.exceptions import UnsupportedInputError

_MAPPERS: list[tuple[type, Callable]] = [
    (AnthropicConfig, anthropic_messages),
    (BedrockConfig, bedrock_messages),
    (DashScopeConfig, dashscope_messages),
    (GeminiConfig, gemini_messages),
    (VertexAIConfig, gemini_messages),
    (MistralConfig, mistral_messages),
    (OllamaConfig, ollama_messages),
    (OpenAIConfig, openai_messages),
    (OpenAIResponsesConfig, responses_messages),
    (XAIConfig, xai_messages),
    (ZAIConfig, zai_messages),
]

_PROBES = [
    BinaryInput(b"abc", media_type="image/png", kind=BinaryType.IMAGE),
    UrlInput("https://example.test/image.png", kind=BinaryType.IMAGE),
    BinaryInput(b"abc", media_type="audio/wav", kind=BinaryType.AUDIO),
    UrlInput("https://example.test/audio.wav", kind=BinaryType.AUDIO),
    BinaryInput(b"abc", media_type="video/mp4", kind=BinaryType.VIDEO),
    UrlInput("https://example.test/video.mp4", kind=BinaryType.VIDEO),
    BinaryInput(b"abc", media_type="application/pdf", kind=BinaryType.DOCUMENT),
    BinaryInput(b"abc", media_type="image/avif", kind=BinaryType.IMAGE),
    BinaryInput(b"abc", media_type="application/zip", kind=BinaryType.DOCUMENT),
    BinaryInput(b"abc", media_type="video/webm", kind=BinaryType.VIDEO),
    UrlInput("https://example.test/file.pdf", kind=BinaryType.DOCUMENT),
    FileIdInput("file-1"),
]


@pytest.mark.parametrize("config_class,mapper", _MAPPERS, ids=lambda value: getattr(value, "__name__", str(value)))
@pytest.mark.parametrize("position", ["user", "tool"])
@pytest.mark.parametrize("part", _PROBES, ids=lambda part: f"{type(part).__name__}-{getattr(part, 'kind', 'file')}")
def test_acceptance_matches_provider_mapper(config_class: type, mapper: Callable, position: str, part: object) -> None:
    config = config_class(model="probe")
    messages = (
        [ModelRequest([part])]
        if position == "user"
        else [
            ModelResponse(tool_calls=ToolCallsEvent([ToolCallEvent(id="call-1", name="lookup", arguments="{}")])),
            ToolResultsEvent([ToolResultEvent(parent_id="call-1", name="lookup", result=ToolResult(parts=[part]))]),
        ]
    )
    try:
        if mapper is responses_messages or mapper in (anthropic_messages, bedrock_messages, gemini_messages):
            mapper(messages, SerializerCls)
        else:
            mapper([], messages, SerializerCls)
        rejected = False
    except UnsupportedInputError:
        rejected = True
    assert accepts_input(config, position, part) is not rejected


@pytest.mark.parametrize("part", _PROBES)
@pytest.mark.parametrize("position", ["user", "tool"])
def test_typesafe_acceptance_matches_its_part_converter(part: object, position: str) -> None:
    try:
        typesafe_parts([part], SerializerCls)
        rejected = False
    except UnsupportedInputError:
        rejected = True
    assert accepts_input(TypeSafeConfig(), position, part) is not rejected


def test_capabilities_use_the_user_position() -> None:
    assert input_modalities(OllamaConfig(model="probe")) == {
        "image": True,
        "audio": False,
        "video": False,
        "pdf": False,
    }
