# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Which media a client may send, per provider, and so what the model is actually handed.

A client can attach any media to a user message or to a tool result. AG-UI passes on only what the
provider behind the run takes in that position; the rest is skipped with a warning and the run
carries on. The table below is the contract, written out per provider so that a wrong entry in the
implementation is caught here rather than copied.
"""

from base64 import b64encode
from collections.abc import Callable
from functools import partial
from typing import Any

import pytest

# This matrix needs every provider installed. The LLM CI matrix installs one at a time.
for module in (
    "anthropic",
    "botocore",
    "dashscope",
    "google.genai",
    "mistralai",
    "ollama",
    "openai",
    "typesafe_sdk",
    "xai_sdk",
    "zai",
):
    pytest.importorskip(module)

from ag_ui.core import (  # noqa: E402
    AssistantMessage,
    AudioPart,
    DataSource,
    DocumentPart,
    FileSource,
    FunctionCall,
    ImagePart,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    TextPart,
    ToolCall,
    ToolMessage,
    UrlSource,
    UserMessage,
    VideoPart,
)
from google.auth.credentials import AnonymousCredentials  # noqa: E402

from ag2 import Agent  # noqa: E402
from ag2.ag_ui import AGUIStream  # noqa: E402
from ag2.config import (  # noqa: E402
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
from ag2.events import (  # noqa: E402
    BinaryInput,
    BinaryType,
    FileIdInput,
    Input,
    ModelRequest,
    ToolResultsEvent,
    UrlInput,
)
from ag2.testing import TestConfig  # noqa: E402
from test.ag_ui.harness import dispatch_events, recording_history, run_input, sole  # noqa: E402

pytestmark = pytest.mark.asyncio

_PROBES: dict[str, Input] = {
    "png": BinaryInput(b"abc", media_type="image/png", kind=BinaryType.IMAGE),
    "image url": UrlInput("https://example.test/image.png", kind=BinaryType.IMAGE),
    "wav": BinaryInput(b"abc", media_type="audio/wav", kind=BinaryType.AUDIO),
    "audio url": UrlInput("https://example.test/audio.wav", kind=BinaryType.AUDIO),
    "mp4": BinaryInput(b"abc", media_type="video/mp4", kind=BinaryType.VIDEO),
    "video url": UrlInput("https://example.test/video.mp4", kind=BinaryType.VIDEO),
    "pdf": BinaryInput(b"abc", media_type="application/pdf", kind=BinaryType.DOCUMENT),
    "avif": BinaryInput(b"abc", media_type="image/avif", kind=BinaryType.IMAGE),
    "zip": BinaryInput(b"abc", media_type="application/zip", kind=BinaryType.DOCUMENT),
    "webm": BinaryInput(b"abc", media_type="video/webm", kind=BinaryType.VIDEO),
    "pdf url": UrlInput("https://example.test/file.pdf", kind=BinaryType.DOCUMENT),
    "file id": FileIdInput("file-1"),
}

_EVERYTHING = set(_PROBES)
_NOTHING: set[str] = set()

_IMAGES_AND_PDF = {"png", "image url", "pdf", "pdf url", "file id"}
_BEDROCK = {"png", "mp4", "pdf", "webm"}
_DASHSCOPE = {"png", "image url", "avif"}
_GEMINI_TOOL_RESULT = {"png", "image url", "pdf", "pdf url", "avif", "zip"}
_MISTRAL = {"png", "image url", "pdf", "pdf url", "avif", "zip", "file id"}
_OPENAI_COMPLETIONS_USER = {"png", "image url", "wav", "pdf", "avif", "zip", "file id"}


def _vertex() -> VertexAIConfig:
    """Vertex has no API key; anonymous credentials keep its client off Google's auth lookup."""
    return VertexAIConfig(model="probe", project="probe", location="us-central1", credentials=AnonymousCredentials())


def _keyed(config_class: type) -> Callable[[], Any]:
    return partial(config_class, model="probe", api_key="test")


def _keyless(config_class: type) -> Callable[[], Any]:
    return partial(config_class, model="probe")


# (a config for the provider, accepted in a user message, accepted in a tool result)
_ACCEPTED: dict[str, tuple[Callable[[], Any], set[str], set[str]]] = {
    "anthropic": (_keyed(AnthropicConfig), _IMAGES_AND_PDF, _IMAGES_AND_PDF),
    "bedrock": (_keyless(BedrockConfig), _BEDROCK, _BEDROCK),
    "dashscope": (_keyed(DashScopeConfig), _DASHSCOPE, _DASHSCOPE),
    "gemini": (_keyed(GeminiConfig), _EVERYTHING, _GEMINI_TOOL_RESULT),
    "vertexai": (_vertex, _EVERYTHING, _GEMINI_TOOL_RESULT),
    "mistral": (_keyed(MistralConfig), _MISTRAL, _MISTRAL),
    "ollama": (_keyless(OllamaConfig), {"png", "avif"}, _NOTHING),
    "openai": (_keyed(OpenAIConfig), _OPENAI_COMPLETIONS_USER, _NOTHING),
    "openai-responses": (_keyed(OpenAIResponsesConfig), _MISTRAL, _MISTRAL),
    "xai": (_keyed(XAIConfig), _MISTRAL, _NOTHING),
    "zai": (_keyed(ZAIConfig), _NOTHING, _NOTHING),
    "typesafe": (_keyed(TypeSafeConfig), _NOTHING, _NOTHING),
}

_CLIENT_PART = {
    BinaryType.IMAGE: ImagePart,
    BinaryType.AUDIO: AudioPart,
    BinaryType.VIDEO: VideoPart,
    BinaryType.DOCUMENT: DocumentPart,
}


def _sent_by_client(probe: Input) -> Any:
    """The part a browser client would send for `probe`."""
    if isinstance(probe, FileIdInput):
        return DocumentPart(source=FileSource(value=probe.file_id))
    if isinstance(probe, BinaryInput):
        source = DataSource(value=b64encode(probe.data).decode(), mime_type=probe.media_type)
    else:
        assert isinstance(probe, UrlInput)
        source = UrlSource(value=probe.url)
    return _CLIENT_PART[probe.kind](source=source)


def _incoming(position: str, part: Any) -> Any:
    if position == "user":
        return run_input(UserMessage(id="u1", content=[TextPart(text="look"), part]))
    return run_input(
        UserMessage(id="u1", content="look"),
        AssistantMessage(id="a1", tool_calls=[ToolCall(id="c1", function=FunctionCall(name="lookup", arguments="{}"))]),
        ToolMessage(id="t1", tool_call_id="c1", content=[TextPart(text="result"), part]),
        UserMessage(id="u2", content="continue"),
    )


@pytest.mark.parametrize("provider", list(_ACCEPTED))
@pytest.mark.parametrize("position", ["user", "tool"])
@pytest.mark.parametrize("probe", list(_PROBES))
async def test_the_model_is_handed_only_the_media_its_provider_takes(provider: str, position: str, probe: str) -> None:
    make_config, in_user, in_tool = _ACCEPTED[provider]
    middleware, calls = recording_history(reply="done")
    agent = Agent("test_agent", config=TestConfig("done"))

    events = await dispatch_events(
        AGUIStream(agent),
        _incoming(position, _sent_by_client(_PROBES[probe])),
        config=make_config(),
        middleware=[middleware],
    )

    assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()
    [call] = calls
    if position == "user":
        [request] = [event for event in call.events if isinstance(event, ModelRequest)]
        handed = _PROBES[probe] in request.parts
    else:
        [results] = [event for event in call.events if isinstance(event, ToolResultsEvent)]
        [result] = results.results
        handed = _PROBES[probe] in result.result.parts
    assert handed is (probe in (in_user if position == "user" else in_tool))


async def test_the_capabilities_advertise_what_a_user_message_may_carry() -> None:
    agent = Agent("test_agent", config=OllamaConfig(model="probe"))

    assert AGUIStream(agent).capabilities().multimodal.input.model_dump() == {
        "image": True,
        "audio": False,
        "video": False,
        "pdf": False,
    }
