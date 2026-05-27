# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultEvent
from autogen.beta.testing import TestConfig
from autogen.beta.tools import SandboxCodeTool
from autogen.beta.tools.code import CodeEnvironment, CodeLanguage, CodeRunResult


class FakeEnv:
    """Tiny in-test backend satisfying the CodeEnvironment protocol."""

    def __init__(
        self,
        *,
        languages: tuple[CodeLanguage, ...] = ("python", "bash"),
        output: str = "",
        exit_code: int = 0,
    ) -> None:
        self._languages = languages
        self._output = output
        self._exit_code = exit_code
        self.calls: list[tuple[str, str]] = []

    @property
    def supported_languages(self) -> tuple[CodeLanguage, ...]:
        return self._languages

    async def run(self, code: str, language: CodeLanguage, *, context=None) -> CodeRunResult:
        self.calls.append((code, language))
        return CodeRunResult(output=self._output, exit_code=self._exit_code)


def _tool_call(code: str, language: str = "python") -> ToolCallEvent:
    return ToolCallEvent(
        arguments=json.dumps({"code": code, "language": language}),
        name="run_code",
    )


def _config(code: str, language: str = "python", final_reply: str = "done") -> TestConfig:
    return TestConfig(
        ModelResponse(tool_calls=ToolCallsEvent([_tool_call(code, language)])),
        final_reply,
    )


class TestSandboxCodeToolConstruction:
    def test_environment_is_required(self) -> None:
        with pytest.raises(TypeError, match="environment"):
            SandboxCodeTool()  # type: ignore[call-arg]

    def test_environment_preserved(self) -> None:
        env = FakeEnv()
        sandbox = SandboxCodeTool(env)
        assert sandbox.environment is env

    @pytest.mark.asyncio
    async def test_supported_languages_in_description(self) -> None:
        env = FakeEnv(languages=("python", "bash"))
        sandbox = SandboxCodeTool(env)
        [schema] = await sandbox.schemas(None)  # type: ignore[arg-type]
        assert "python" in schema.function.description
        assert "bash" in schema.function.description

    def test_custom_name_used(self) -> None:
        sandbox = SandboxCodeTool(FakeEnv(), name="my_runner")
        assert sandbox.name == "my_runner"


class TestSandboxCodeToolExecution:
    @pytest.mark.asyncio
    async def test_success_propagates_output(self) -> None:
        results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: results.append(str(e.result)))

        env = FakeEnv(output="42", exit_code=0)
        agent = Agent("a", config=_config("print(42)"), tools=[SandboxCodeTool(env)])
        await agent.ask("compute", stream=stream)

        assert results
        assert "42" in results[0]

    @pytest.mark.asyncio
    async def test_failure_includes_exit_code(self) -> None:
        results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: results.append(str(e.result)))

        env = FakeEnv(output="boom", exit_code=7)
        agent = Agent("a", config=_config("import sys; sys.exit(7)"), tools=[SandboxCodeTool(env)])
        await agent.ask("fail", stream=stream)

        assert results
        assert "exit code: 7" in results[0]

    @pytest.mark.asyncio
    async def test_success_omits_exit_code(self) -> None:
        results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: results.append(str(e.result)))

        env = FakeEnv(output="ok", exit_code=0)
        agent = Agent("a", config=_config("print('ok')"), tools=[SandboxCodeTool(env)])
        await agent.ask("run", stream=stream)

        assert results
        assert "exit code" not in results[0]

    @pytest.mark.asyncio
    async def test_environment_receives_call(self) -> None:
        env = FakeEnv(output="fake output", exit_code=0)
        agent = Agent("a", config=_config("print(1)"), tools=[SandboxCodeTool(env)])
        await agent.ask("run")

        assert env.calls == [("print(1)", "python")]


class TestSandboxCodeToolWithCustomEnvironment:
    """Verifies the tool only depends on the CodeEnvironment protocol —
    any object satisfying it works.
    """

    @pytest.mark.asyncio
    async def test_arbitrary_environment_is_invoked(self) -> None:
        captured: list[tuple[str, str]] = []

        class CustomEnv:
            @property
            def supported_languages(self) -> tuple[CodeLanguage, ...]:
                return ("python",)

            async def run(self, code: str, language: CodeLanguage, *, context=None) -> CodeRunResult:
                captured.append((code, language))
                return CodeRunResult(output="custom output", exit_code=0)

        env: CodeEnvironment = CustomEnv()
        sandbox = SandboxCodeTool(env)

        results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: results.append(str(e.result)))

        agent = Agent("a", config=_config("print(1)"), tools=[sandbox])
        await agent.ask("run", stream=stream)

        assert captured == [("print(1)", "python")]
        assert results
        assert "custom output" in results[0]
