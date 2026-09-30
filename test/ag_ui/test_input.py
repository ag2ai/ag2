# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""What a `RunAgentInput` becomes by the time the model is called: history, current turn, prompt."""

import json
import logging
from base64 import b64encode
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from ag_ui.core import (
    AssistantMessage,
    AudioPart,
    DataSource,
    DocumentPart,
    FileSource,
    FunctionCall,
    ImagePart,
    ReasoningMessage,
    RunAgentInput,
    RunFinishedEvent,
    RunFinishedSuccessOutcome,
    SystemMessage,
    TextPart,
    ToolCall,
    ToolMessage,
    UrlSource,
    UserMessage,
    VideoPart,
)
from ag_ui.core import Context as ContextEntry

from ag2 import Agent, Context, ToolResult
from ag2.ag_ui import AGUIStream, read_run_input
from ag2.config import ModelProvider, OpenAIConfig
from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.events import (
    AudioInput,
    BaseEvent,
    BinaryInput,
    BinaryType,
    DocumentInput,
    FileIdInput,
    ImageInput,
    ModelMessage,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
    VideoInput,
)
from ag2.testing import TestConfig
from test.ag_ui.harness import dispatch_events, recording_history, run_input, sole

pytestmark = pytest.mark.asyncio

RAW_BYTES = b"\xff\xd8\xff\xe0"
B64_VALUE = b64encode(RAW_BYTES).decode()


@dataclass
class ModelInput:
    """What the model was handed: the prompt, the history before this turn, and this turn's inputs."""

    prompt: list[str]
    history: list[BaseEvent]
    turn: list[Any]
    events: list[Any]


async def _model_input(incoming: RunAgentInput, *, config: Any = None) -> ModelInput:
    """Serve `incoming` and report what reached the model. `config` is the run's own, as a client would pick it."""
    middleware, calls = recording_history(reply="done")
    agent = Agent("test_agent", config=TestConfig("done"))

    events = await dispatch_events(
        AGUIStream(agent), incoming, middleware=[middleware], **({"config": config} if config else {})
    )

    [call] = calls
    *history, last = call.events
    if isinstance(last, ModelRequest):
        return ModelInput(call.prompt, history, list(last.parts), events)
    return ModelInput(call.prompt, call.events, [], events)


def _as(provider: ModelProvider) -> TestConfig:
    return TestConfig("done", provider=provider)


class TestUserMessages:
    async def test_a_plain_string_becomes_text_input(self) -> None:
        model = await _model_input(run_input(UserMessage(id="m1", content="hello")))

        assert (model.prompt, model.history, model.turn) == ([], [], [TextInput("hello")])

    async def test_text_content_becomes_text_input(self) -> None:
        model = await _model_input(run_input(UserMessage(id="m1", content=[TextPart(text="hi")])))

        assert (model.history, model.turn) == ([], [TextInput("hi")])

    async def test_a_system_message_goes_to_the_prompt(self) -> None:
        model = await _model_input(
            run_input(SystemMessage(id="s1", content="be brief"), UserMessage(id="u1", content="hi"))
        )

        assert (model.prompt, model.history, model.turn) == (["be brief"], [], [TextInput("hi")])

    async def test_trailing_user_messages_are_split_from_history(self) -> None:
        model = await _model_input(
            run_input(
                UserMessage(id="u1", content="first turn"),
                AssistantMessage(id="a1", content="reply"),
                UserMessage(id="u2", content="follow-up"),
            )
        )

        assert model.history == [
            ModelRequest([TextInput("first turn")]),
            ModelResponse(ModelMessage("reply"), tool_calls=ToolCallsEvent([])),
        ]
        assert model.turn == [TextInput("follow-up")]

    async def test_consecutive_trailing_user_messages_make_one_turn(self) -> None:
        model = await _model_input(
            run_input(UserMessage(id="u1", content="hello"), UserMessage(id="u2", content="and more"))
        )

        assert (model.history, model.turn) == ([], [TextInput("hello"), TextInput("and more")])


@pytest.mark.parametrize(
    "content_cls,factory,kind,mime",
    [
        (ImagePart, ImageInput, BinaryType.IMAGE, "image/jpeg"),
        (AudioPart, AudioInput, BinaryType.AUDIO, "audio/wav"),
        (VideoPart, VideoInput, BinaryType.VIDEO, "video/mp4"),
        (DocumentPart, DocumentInput, BinaryType.DOCUMENT, "application/pdf"),
    ],
)
class TestMediaParts:
    async def test_a_url_source_becomes_a_url_input(
        self, content_cls: type, factory: Any, kind: BinaryType, mime: str
    ) -> None:
        url = "https://example.com/file"
        incoming = run_input(UserMessage(id="m1", content=[content_cls(source=UrlSource(value=url))]))

        model = await _model_input(incoming)

        assert model.turn == [factory(url)]
        assert isinstance(model.turn[0], UrlInput)
        assert model.turn[0].kind == kind

    async def test_a_data_source_becomes_a_binary_input(
        self, content_cls: type, factory: Any, kind: BinaryType, mime: str
    ) -> None:
        incoming = run_input(
            UserMessage(id="m1", content=[content_cls(source=DataSource(value=B64_VALUE, mime_type=mime))])
        )

        model = await _model_input(incoming)

        assert model.turn == [factory(data=RAW_BYTES, media_type=mime)]
        assert isinstance(model.turn[0], BinaryInput)
        assert model.turn[0].kind == kind


class TestPartMetadata:
    async def test_it_propagates_to_the_input(self) -> None:
        incoming = run_input(
            UserMessage(
                id="m1",
                content=[
                    TextPart(text="hi"),
                    ImagePart(source=UrlSource(value="https://x/i.png"), metadata={"alt": "cat"}),
                ],
            )
        )

        model = await _model_input(incoming)

        assert [part.metadata for part in model.turn] == [{}, {"alt": "cat"}]

    async def test_it_survives_base64_decoding(self) -> None:
        """A part carried as data, not as a URL, keeps its metadata through the decode."""
        incoming = run_input(
            UserMessage(
                id="m1",
                content=[
                    DocumentPart(
                        source=DataSource(value=B64_VALUE, mime_type="application/pdf"),
                        metadata={"source_filename": "report.pdf"},
                    )
                ],
            )
        )

        model = await _model_input(incoming)

        expected = DocumentInput(data=RAW_BYTES, media_type="application/pdf")
        expected.metadata = {"source_filename": "report.pdf"}
        assert model.turn == [expected]


class TestHistoryRoles:
    async def test_an_assistant_message_becomes_a_model_response(self) -> None:
        model = await _model_input(
            run_input(UserMessage(id="u1", content="hi"), AssistantMessage(id="a1", content="hello!"))
        )

        assert model.turn == []
        assert model.history == [
            ModelRequest([TextInput("hi")]),
            ModelResponse(ModelMessage("hello!"), tool_calls=ToolCallsEvent([])),
        ]

    async def test_a_tool_message_becomes_the_result_of_the_call_it_answers(self) -> None:
        incoming = run_input(
            UserMessage(id="u1", content="run tool"),
            AssistantMessage(
                id="a1",
                content=None,
                tool_calls=[ToolCall(id="t1", type="function", function=FunctionCall(name="do", arguments="{}"))],
            ),
            ToolMessage(id="tm1", tool_call_id="t1", content="42"),
        )

        model = await _model_input(incoming)

        assert model.turn == []
        assert model.history == [
            ModelRequest([TextInput("run tool")]),
            ModelResponse(None, tool_calls=ToolCallsEvent([ToolCallEvent(id="t1", name="do", arguments="{}")])),
            ToolResultsEvent([ToolResultEvent(parent_id="t1", name="do", result=ToolResult("42"))]),
        ]

    async def test_a_tool_message_answering_no_restated_call_stays_unnamed(self) -> None:
        model = await _model_input(run_input(ToolMessage(id="tm1", tool_call_id="elsewhere", content="sunny")))

        assert model.history == [ToolResultsEvent([ToolResultEvent(parent_id="elsewhere", result=ToolResult("sunny"))])]

    async def test_a_tool_message_in_parts_becomes_a_tool_result_of_those_parts(self) -> None:
        """A screenshot beside its caption reaches the model as both, not as a string."""
        incoming = run_input(
            ToolMessage(
                id="tm1",
                tool_call_id="t1",
                content=[
                    TextPart(text="the page"),
                    ImagePart(source=DataSource(value=B64_VALUE, mime_type="image/jpeg")),
                ],
            )
        )

        model = await _model_input(incoming)

        assert model.history == [
            ToolResultsEvent([
                ToolResultEvent(
                    parent_id="t1",
                    result=ToolResult(TextInput("the page"), ImageInput(data=RAW_BYTES, media_type="image/jpeg")),
                )
            ])
        ]

    async def test_a_tool_message_s_error_leads_and_its_content_is_kept(self) -> None:
        incoming = run_input(
            ToolMessage(id="tm1", tool_call_id="t1", content=[TextPart(text="partial")], error="it broke")
        )

        model = await _model_input(incoming)

        # The error leads, and what came with it is kept: a partial result survives.
        assert model.history == [
            ToolResultsEvent([
                ToolResultEvent(parent_id="t1", result=ToolResult(parts=[TextInput("it broke"), TextInput("partial")]))
            ])
        ]

    async def test_a_reasoning_message_is_history_not_part_of_this_turn(self) -> None:
        incoming = run_input(
            UserMessage(id="u1", content="Hi"), ReasoningMessage(id="r1", content="user is greeting me")
        )

        model = await _model_input(incoming)

        assert model.turn == []
        assert model.history == [ModelRequest([TextInput("Hi")]), ModelReasoning("user is greeting me")]

    async def test_an_empty_reasoning_message_is_dropped(self) -> None:
        """Rather than handed to the LLM as a thought with nothing in it."""
        incoming = run_input(UserMessage(id="u1", content="Hi"), ReasoningMessage(id="r1", content=""))

        model = await _model_input(incoming)

        assert model.history == [ModelRequest([TextInput("Hi")])]


async def test_gemini_restated_tool_call_keeps_its_signature() -> None:
    incoming = run_input(
        AssistantMessage(
            id="a1",
            tool_calls=[
                ToolCall(id="call-1", function=FunctionCall(name="lookup", arguments="{}"), encrypted_value=B64_VALUE)
            ],
        )
    )

    gemini = await _model_input(incoming, config=_as(ModelProvider.GEMINI))
    openai = await _model_input(incoming, config=_as(ModelProvider.OPENAI))

    [call] = gemini.history[0].tool_calls.calls
    assert isinstance(call, GeminiToolCallEvent)
    assert call.thought_signature == RAW_BYTES
    [other] = openai.history[0].tool_calls.calls
    assert type(other) is ToolCallEvent


class TestPartsTheProviderCannotTake:
    async def test_an_unsupported_user_part_is_skipped_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        incoming = run_input(
            UserMessage(
                id="m1",
                content=[TextPart(text="hello"), VideoPart(source=DataSource(value=B64_VALUE, mime_type="video/mp4"))],
            )
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(incoming, config=OpenAIConfig(model="gpt-4o", api_key="test"))

        assert model.turn == [TextInput("hello")]
        [record] = caplog.records
        assert "video" in record.getMessage()
        assert "video/mp4" in record.getMessage()
        assert B64_VALUE not in caplog.text

    async def test_an_unsupported_tool_part_still_answers_with_empty_text(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        incoming = run_input(
            AssistantMessage(
                id="a1", tool_calls=[ToolCall(id="call-1", function=FunctionCall(name="lookup", arguments="{}"))]
            ),
            ToolMessage(
                id="t1",
                tool_call_id="call-1",
                content=[ImagePart(source=DataSource(value=B64_VALUE, mime_type="image/png"))],
            ),
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(incoming, config=OpenAIConfig(model="gpt-4o", api_key="test"))

        [results] = [event for event in model.history if isinstance(event, ToolResultsEvent)]
        assert [r.result.parts for r in results.results] == [[TextInput("")]]
        assert len(caplog.records) == 1


class TestProviderFileHandles:
    """A handle only the provider that minted it can resolve, and never fetched or parsed."""

    async def test_an_untagged_handle_is_taken_to_be_the_run_s_own(self) -> None:
        incoming = run_input(UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc"))]))

        model = await _model_input(incoming, config=_as(ModelProvider.ANTHROPIC))

        assert model.turn == [FileIdInput("file-abc")]

    async def test_a_handle_tagged_with_the_run_s_provider_reaches_it(self) -> None:
        incoming = run_input(
            UserMessage(id="m1", content=[ImagePart(source=FileSource(value="file-abc", provider="anthropic"))])
        )

        model = await _model_input(incoming, config=_as(ModelProvider.ANTHROPIC))

        assert model.turn == [FileIdInput("file-abc")]

    @pytest.mark.parametrize(
        ("config", "tag"),
        [
            (ModelProvider.GEMINI, "google"),
            (ModelProvider.VERTEXAI, "google"),
            (ModelProvider.GEMINI, "gemini"),
        ],
    )
    async def test_a_handle_tagged_with_the_vendor_reaches_the_run(self, config: ModelProvider, tag: str) -> None:
        """The protocol names a provider by vendor (`google`); ag2's names are for the API."""
        incoming = run_input(
            UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="files/abc", provider=tag))])
        )

        model = await _model_input(incoming, config=_as(config))

        assert model.turn == [FileIdInput("files/abc")]

    async def test_another_provider_s_handle_is_skipped_and_said_so(self, caplog: pytest.LogCaptureFixture) -> None:
        incoming = run_input(
            UserMessage(
                id="m1",
                content=[
                    TextPart(text="look"),
                    DocumentPart(source=FileSource(value="file-secret", provider="openai")),
                ],
            )
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(incoming, config=_as(ModelProvider.ANTHROPIC))

        assert model.turn == [TextInput("look")]
        assert sole(model.events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()
        [record] = caplog.records
        assert "openai" in record.getMessage()
        assert "anthropic" in record.getMessage()
        assert "file-secret" not in record.getMessage()

    async def test_the_config_the_run_is_served_with_decides_the_provider(self) -> None:
        """Not the agent's default: a handle minted for OpenAI reaches an OpenAI-configured run."""
        incoming = run_input(
            UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc", provider="openai"))])
        )

        model = await _model_input(incoming, config=_as(ModelProvider.OPENAI))

        assert model.turn == [FileIdInput("file-abc")]

    async def test_a_skipped_handle_in_a_tool_message_leaves_the_rest_of_the_result(self) -> None:
        incoming = run_input(
            ToolMessage(
                id="tm1",
                tool_call_id="t1",
                content=[TextPart(text="done"), DocumentPart(source=FileSource(value="f", provider="openai"))],
            )
        )

        model = await _model_input(incoming, config=_as(ModelProvider.GEMINI))

        assert model.history == [ToolResultsEvent([ToolResultEvent(parent_id="t1", result=ToolResult("done"))])]

    async def test_a_tool_answer_left_with_no_parts_reaches_the_model_as_the_empty_string(self) -> None:
        """The call is still answered: a result whose every part was dropped is `""`."""
        incoming = run_input(
            UserMessage(id="m1", content="read the report"),
            AssistantMessage(
                id="m2",
                tool_calls=[ToolCall(id="c1", type="function", function=FunctionCall(name="fetch", arguments="{}"))],
            ),
            ToolMessage(
                id="m3",
                tool_call_id="c1",
                content=[DocumentPart(source=FileSource(value="file-abc", provider="openai"))],
            ),
        )

        model = await _model_input(incoming, config=_as(ModelProvider.ANTHROPIC))

        [results] = [event for event in model.history if isinstance(event, ToolResultsEvent)]
        assert [r.result for r in results.results] == [ToolResult(TextInput(""))]


async def test_a_run_of_only_thread_run_and_messages_is_served() -> None:
    """`tools`, `context` and `forwardedProps` are optional in 1.0, and absent means none."""
    agent = Agent("test_agent", config=TestConfig("hello"))
    incoming = RunAgentInput.model_validate({
        "threadId": "t1",
        "runId": "r1",
        "messages": [{"id": "m1", "role": "user", "content": "hi"}],
    })

    events = await dispatch_events(AGUIStream(agent), incoming)

    assert sole(events, RunFinishedEvent).outcome == RunFinishedSuccessOutcome()


def _body(*messages: dict[str, Any], **extra: Any) -> str:
    return json.dumps({"threadId": "t1", "runId": "r1", "protocolVersion": "1.0", "messages": list(messages), **extra})


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("ag2.ag_ui") and r.levelno == logging.WARNING]


class TestUnrecognisedInput:
    """Input material this server does not recognise is stripped with a warning, and the run is served.

    A malformed *known* value still refuses the input; see `served/test_run_input_parsing.py`.
    """

    async def test_a_part_of_an_unknown_kind_is_stripped_and_the_rest_reaches_the_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _body({
            "id": "m1",
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "hologram", "value": "?"}],
        })

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(read_run_input(body))

        assert model.turn == [TextInput("look")]
        [warning] = _warnings(caplog)
        assert "/messages/0/content/1" in warning

    async def test_an_unknown_top_level_property_is_stripped_and_the_run_served(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _body({"id": "m1", "role": "user", "content": "hi"}, fromTheFuture=True)

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(read_run_input(body))

        assert model.turn == [TextInput("hi")]
        [warning] = _warnings(caplog)
        assert "/fromTheFuture" in warning

    async def test_an_unknown_nested_property_is_stripped(self, caplog: pytest.LogCaptureFixture) -> None:
        body = _body({"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi", "glow": "blue"}]})

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            incoming = read_run_input(body)

        assert incoming.model_dump(by_alias=True, exclude_none=True)["messages"] == [
            {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        [warning] = _warnings(caplog)
        assert "/messages/0/content/0/glow" in warning

    async def test_a_message_of_an_unknown_role_is_stripped(self, caplog: pytest.LogCaptureFixture) -> None:
        body = _body({"id": "m0", "role": "oracle", "content": "?"}, {"id": "m1", "role": "user", "content": "hi"})

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(read_run_input(body))

        assert (model.history, model.turn) == ([], [TextInput("hi")])
        [warning] = _warnings(caplog)
        assert "/messages/0" in warning

    async def test_open_objects_keep_what_the_protocol_does_not_describe(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`state`, `forwardedProps` and `metadata` are open by key: nothing in them is unrecognised."""
        body = _body(
            {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi", "metadata": {"mine": None}}]},
            state={"draft": {"anything": None}},
            forwardedProps={"app": [1, None]},
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            incoming = read_run_input(body)

        assert (incoming.state, incoming.forwarded_props) == ({"draft": {"anything": None}}, {"app": [1, None]})
        assert _warnings(caplog) == []

    async def test_a_part_whose_source_is_of_an_unknown_kind_is_stripped_whole(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A part left without its source would be malformed, so the whole part goes."""
        body = _body({
            "id": "m1",
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "ipfs", "value": "Qm..."}},
            ],
        })

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(read_run_input(body))

        assert model.turn == [TextInput("look")]
        [warning] = _warnings(caplog)
        assert "/messages/0/content/1" in warning

    async def test_a_tool_message_whose_only_part_is_stripped_is_answered_with_the_empty_string(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        body = _body(
            {"id": "m1", "role": "user", "content": "weather?"},
            {
                "id": "a1",
                "role": "assistant",
                "toolCalls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {
                "id": "t1",
                "role": "tool",
                "toolCallId": "c1",
                "content": [{"type": "image", "source": {"type": "ipfs", "value": "Qm..."}}],
            },
        )

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            model = await _model_input(read_run_input(body))

        assert [event for event in model.history if isinstance(event, ToolResultsEvent)] == [
            ToolResultsEvent([ToolResultEvent(parent_id="c1", name="get_weather", result=ToolResult(""))])
        ]
        [warning] = _warnings(caplog)
        assert "/messages/2/content/0" in warning


_ENTRIES = [
    ContextEntry(description="The user's locale", value="en-GB"),
    ContextEntry(description="Open document", value='{"title": "Q3 plan"}'),
]


def _prompt_watcher() -> tuple[Agent, list[list[str]]]:
    """An agent whose one tool records the prompt it runs under."""
    seen: list[list[str]] = []
    agent = Agent("test_agent", prompt="Be helpful.", config=TestConfig(ToolCallEvent(name="peek"), "done"))

    @agent.tool
    def peek(context: Context) -> str:
        """Look at the prompt."""
        seen.append(list(context.prompt))
        return "seen"

    return agent, seen


def _as_pairs(entries: Sequence[ContextEntry]) -> str:
    return "\n".join(f"{e.description}={e.value}" for e in entries)


class TestClientContext:
    """A run input's `context` entries are made available to the model, as part of its prompt."""

    async def test_each_entry_reaches_the_prompt(self) -> None:
        agent, seen = _prompt_watcher()
        incoming = run_input(UserMessage(id="m1", content="hi"))
        incoming.context = _ENTRIES

        await dispatch_events(AGUIStream(agent), incoming)

        assert seen == [
            [
                "Be helpful.",
                '## Context from the application\n\n- The user\'s locale: en-GB\n- Open document: {"title": "Q3 plan"}',
            ]
        ]

    async def test_without_context_the_prompt_is_unchanged(self) -> None:
        agent, seen = _prompt_watcher()

        await dispatch_events(AGUIStream(agent), run_input(UserMessage(id="m1", content="hi")))

        assert seen == [["Be helpful."]]

    async def test_the_application_can_turn_the_context_block_off(self) -> None:
        agent, seen = _prompt_watcher()
        incoming = run_input(UserMessage(id="m1", content="hi"))
        incoming.context = _ENTRIES

        await dispatch_events(AGUIStream(agent), incoming, context_prompt=None)

        assert seen == [["Be helpful."]]

    async def test_the_application_can_render_the_context_its_own_way(self) -> None:
        agent, seen = _prompt_watcher()
        incoming = run_input(UserMessage(id="m1", content="hi"))
        incoming.context = _ENTRIES

        await dispatch_events(AGUIStream(agent), incoming, context_prompt=_as_pairs)

        assert seen == [["Be helpful.", 'The user\'s locale=en-GB\nOpen document={"title": "Q3 plan"}']]
