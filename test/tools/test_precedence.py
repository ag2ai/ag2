# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from fast_depends.library.serializer import SerializerProto
from typing_extensions import Self

from ag2 import Agent, Context, Toolkit, tool
from ag2.config import LLMClient, ModelConfig, ModelProvider
from ag2.context import ConversationContext
from ag2.events import BaseEvent, ModelResponse, ToolCallEvent
from ag2.middleware import approval_required
from ag2.response import ResponseProto
from ag2.testing import TestConfig
from ag2.tools import AnthropicBashTool
from ag2.tools.schemas import ToolSchema


class SchemaRecordingConfig(ModelConfig):
    """``TestConfig`` that also keeps the tool schemas each LLM call was offered."""

    def __init__(self, *turns: Any) -> None:
        self.config = TestConfig(*turns)
        self.offered: list[list[ToolSchema]] = []

    @property
    def provider(self) -> ModelProvider:
        return ModelProvider.OPENAI

    @property
    def model(self) -> str:
        return "test-model"

    def copy(self) -> Self:
        return self

    def create(self) -> LLMClient:
        return SchemaRecordingClient(self.config, self.offered)

    def create_files_client(self) -> Any:
        raise NotImplementedError


class SchemaRecordingClient:
    def __init__(self, config: TestConfig, offered: list[list[ToolSchema]]) -> None:
        self.client = config.create()
        self.offered = offered

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: ConversationContext,
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        schemas = list(tools)
        self.offered.append(schemas)
        return await self.client(
            messages, context, tools=schemas, response_schema=response_schema, serializer=serializer
        )


@pytest.mark.asyncio
class TestToolsDeclaredInCode:
    async def test_later_tool_overrides_an_earlier_one_with_a_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        runs: list[str] = []

        def deploy() -> str:
            runs.append("toolkit")
            return "deployed"

        @tool(name="deploy")
        def my_deploy() -> str:
            runs.append("mine")
            return "deployed"

        agent = Agent(
            "",
            tools=[Toolkit(deploy), my_deploy],
            config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
        )

        with caplog.at_level("WARNING", logger="ag2.tools.precedence"):
            await agent.ask("Deploy")

        assert runs == ["mine"]
        [warning] = caplog.records
        assert "deploy" in warning.getMessage()

    async def test_the_same_tool_passed_twice_runs_once(self) -> None:
        runs: list[str] = []

        @tool
        def deploy() -> str:
            runs.append("deploy")
            return "deployed"

        agent = Agent(
            "",
            tools=[deploy, deploy],
            config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
        )

        await agent.ask("Deploy")

        assert runs == ["deploy"]

    async def test_denial_is_not_bypassed_by_an_earlier_ungated_tool(self) -> None:
        runs: list[str] = []

        @tool(name="deploy")
        def ungated() -> str:
            runs.append("ungated")
            return "deployed"

        @tool(name="deploy", middleware=[approval_required()])
        def gated() -> str:
            runs.append("gated")
            return "deployed"

        agent = Agent(
            "",
            tools=[ungated, gated],
            config=TestConfig(ToolCallEvent(name="deploy", arguments="{}"), "done"),
        )

        await agent.ask("Deploy", hitl_hook=lambda _: "n")

        assert runs == []

    @pytest.mark.parametrize("builtin_first", [True, False], ids=["builtin-first", "function-first"])
    async def test_function_and_builtin_sharing_a_name_resolve_by_order(
        self,
        builtin_first: bool,
        context: Context,
    ) -> None:
        @tool
        def bash(command: str) -> str:
            return command

        builtin = AnthropicBashTool()
        tools = [builtin, bash] if builtin_first else [bash, builtin]
        config = SchemaRecordingConfig("done")

        await Agent("", tools=tools, config=config).ask("Run it")

        assert config.offered == [list(await tools[-1].schemas(context))]
