# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Every input AG-UI 1.0 considers valid is served."""

import logging
from collections.abc import Sequence

import pytest
from ag_ui.core import (
    AssistantMessage,
    DocumentPart,
    FileSource,
    FunctionCall,
    RunAgentInput,
    TextPart,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from ag2 import Agent, Context, ToolResult
from ag2.ag_ui import AGUIStream
from ag2.config import ModelProvider
from ag2.events import BaseEvent, FileIdInput, ModelRequest, ModelResponse, TextInput, ToolResultsEvent
from ag2.middleware import BaseMiddleware, LLMCall, Middleware
from ag2.testing import TestConfig, TrackingConfig
from test.ag_ui.harness import dispatch_run, outcome_of, run_input

pytestmark = pytest.mark.asyncio


async def test_a_run_of_only_thread_run_and_messages_is_served() -> None:
    """`tools`, `context` and `forwardedProps` are optional in 1.0, and absent means none."""
    agent = Agent("test_agent", config=TestConfig("hello"))
    incoming = RunAgentInput.model_validate({
        "threadId": "t1",
        "runId": "r1",
        "messages": [{"id": "m1", "role": "user", "content": "hi"}],
    })

    events = await dispatch_run(AGUIStream(agent), incoming)

    assert outcome_of(events) == {"type": "success"}


class TestProviderFileHandles:
    def _agent(self, provider: ModelProvider) -> tuple[Agent, TrackingConfig]:
        tracking = TrackingConfig(TestConfig("read it", provider=provider))
        return Agent("test_agent", config=tracking), tracking

    async def test_an_untagged_handle_reaches_the_model(self) -> None:
        agent, tracking = self._agent(ModelProvider.ANTHROPIC)

        await dispatch_run(
            AGUIStream(agent),
            run_input(UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc"))])),
        )

        [(sent,)] = [call.args for call in tracking.mock.call_args_list]
        assert sent == ModelRequest([FileIdInput("file-abc")])

    async def test_a_handle_tagged_with_the_agent_s_provider_reaches_the_model(self) -> None:
        agent, tracking = self._agent(ModelProvider.ANTHROPIC)

        await dispatch_run(
            AGUIStream(agent),
            run_input(
                UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc", provider="anthropic"))])
            ),
        )

        [(sent,)] = [call.args for call in tracking.mock.call_args_list]
        assert sent == ModelRequest([FileIdInput("file-abc")])

    async def test_another_provider_s_handle_is_skipped_and_the_run_completes(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        agent, tracking = self._agent(ModelProvider.ANTHROPIC)

        with caplog.at_level(logging.WARNING, logger="ag2.ag_ui"):
            events = await dispatch_run(
                AGUIStream(agent),
                run_input(
                    UserMessage(
                        id="m1",
                        content=[
                            TextPart(text="summarise"),
                            DocumentPart(source=FileSource(value="file-abc", provider="openai")),
                        ],
                    )
                ),
            )

        assert outcome_of(events) == {"type": "success"}
        [(sent,)] = [call.args for call in tracking.mock.call_args_list]
        assert sent == ModelRequest([TextInput("summarise")])
        [warning] = caplog.records
        assert "openai" in warning.getMessage()
        assert "anthropic" in warning.getMessage()

    async def test_the_config_passed_to_dispatch_is_the_run_s_provider(self) -> None:
        """The run's provider is the configuration the run is served with, not the agent's default."""
        agent, _ = self._agent(ModelProvider.ANTHROPIC)
        tracking = TrackingConfig(TestConfig("read it", provider=ModelProvider.OPENAI))

        await dispatch_run(
            AGUIStream(agent),
            run_input(
                UserMessage(id="m1", content=[DocumentPart(source=FileSource(value="file-abc", provider="openai"))])
            ),
            config=tracking,
        )

        [(sent,)] = [call.args for call in tracking.mock.call_args_list]
        assert sent == ModelRequest([FileIdInput("file-abc")])

    async def test_a_tool_answer_left_with_no_parts_reaches_the_model_as_the_empty_string(self) -> None:
        """The call is still answered: a result whose every part was dropped is `""`."""
        agent, _ = self._agent(ModelProvider.ANTHROPIC)
        seen: list[Sequence[BaseEvent]] = []

        await dispatch_run(
            AGUIStream(agent),
            run_input(
                UserMessage(id="m1", content="read the report"),
                AssistantMessage(
                    id="m2",
                    tool_calls=[
                        ToolCall(id="c1", type="function", function=FunctionCall(name="fetch", arguments="{}"))
                    ],
                ),
                ToolMessage(
                    id="m3",
                    tool_call_id="c1",
                    content=[DocumentPart(source=FileSource(value="file-abc", provider="openai"))],
                ),
            ),
            middleware=[Middleware(_Recorder, seen=seen)],
        )

        [results] = [event for event in seen[0] if isinstance(event, ToolResultsEvent)]
        assert [r.result for r in results.results] == [ToolResult(TextInput(""))]


class _Recorder(BaseMiddleware):
    """Keeps every history the model is called with."""

    def __init__(self, event: BaseEvent, context: Context, *, seen: list[Sequence[BaseEvent]]) -> None:
        super().__init__(event, context)
        self._seen = seen

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        self._seen.append(list(events))
        return await call_next(events, context)
