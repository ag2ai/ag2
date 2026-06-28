# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import pytest

from ag2 import Agent, MemoryStream
from ag2.events import ModelResponse, ToolCallEvent, ToolCallsEvent, ToolResultEvent
from ag2.testing import TestConfig
from ag2.tools import LocalEnvironment, SandboxShellTool
from ag2.tools.sandbox import ExecResult, SandboxFactory
from ag2.tools.sandbox.filter import check_ignore, matches


class TestMatches:
    def test_plain_prefix_matches(self) -> None:
        assert matches("git", "git status") is True

    def test_plain_prefix_no_match(self) -> None:
        assert matches("git", "rm -rf /") is False

    def test_multi_word_prefix(self) -> None:
        assert matches("uv run", "uv run pytest") is True
        assert matches("uv run", "uv add requests") is False

    def test_rm_rf_blocked(self) -> None:
        assert matches("rm -rf", "rm -rf /") is True
        assert matches("rm -rf", "rm file.txt") is False

    def test_leading_whitespace_stripped(self) -> None:
        assert matches("git", "  git status") is True

    def test_exact_command_matches(self) -> None:
        # "git" alone (no args) should match
        assert matches("git", "git") is True

    def test_word_boundary_no_false_positive(self) -> None:
        # "git" must not match "gitconfig" or "gitfoo"
        assert matches("git", "gitconfig --list") is False
        assert matches("cat", "catchphrase") is False
        assert matches("py", "python3 app.py") is False


class TestCheckIgnore:
    def test_env_file_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat .env", tmp_path, ["**/.env"])
        assert result is not None
        assert ".env" in result

    def test_key_file_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat server.key", tmp_path, ["*.key"])
        assert result is not None
        assert "server.key" in result

    def test_secrets_dir_blocked(self, tmp_path: Path) -> None:
        result = check_ignore("cat secrets/db.key", tmp_path, ["secrets/**"])
        assert result is not None
        assert "secrets" in result

    def test_safe_file_allowed(self, tmp_path: Path) -> None:
        result = check_ignore("cat app.py", tmp_path, ["**/.env", "*.key"])
        assert result is None

    def test_quoted_path_handled(self, tmp_path: Path) -> None:
        result = check_ignore('cat ".env"', tmp_path, ["**/.env"])
        assert result is not None

    def test_plain_filename_blocked(self, tmp_path: Path) -> None:
        assert check_ignore("cat .env", tmp_path, [".env"]) is not None

    def test_plain_dirname_blocks_contents(self, tmp_path: Path) -> None:
        assert check_ignore("cat secrets/db.key", tmp_path, ["secrets"]) is not None
        assert check_ignore("cat secrets/nested/x.txt", tmp_path, ["secrets"]) is not None
        assert check_ignore("cat config/prod.yaml", tmp_path, ["secrets"]) is None

    def test_no_patterns_returns_none(self, tmp_path: Path) -> None:
        assert check_ignore("cat .env", tmp_path, []) is None

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        # ../../../etc/passwd resolves outside workdir — must be denied
        result = check_ignore("cat ../../../etc/passwd", tmp_path, ["**/.env"])
        assert result is not None
        assert "Access denied" in result

    def test_absolute_path_outside_workdir_blocked(self, tmp_path: Path) -> None:
        # Absolute path outside workdir must be denied regardless of patterns
        result = check_ignore("cat /etc/passwd", tmp_path, ["**/.env"])
        assert result is not None
        assert "Access denied" in result

    def test_remote_pure_workdir_blocks_matching_path(self) -> None:
        # Finding #2: a remote backend has no host filesystem — paths are
        # checked lexically against a PurePosixPath workdir, no .resolve().
        result = check_ignore("cat .env", PurePosixPath("/workspace"), ["**/.env"])
        assert result is not None
        assert ".env" in result

    def test_remote_pure_workdir_allows_safe_path(self) -> None:
        assert check_ignore("cat README.md", PurePosixPath("/workspace"), ["**/.env"]) is None

    def test_remote_pure_workdir_blocks_traversal(self) -> None:
        result = check_ignore("cat ../../etc/passwd", PurePosixPath("/workspace"), ["**/.env"])
        assert result is not None
        assert "Access denied" in result

    def test_remote_pure_workdir_blocks_absolute_outside(self) -> None:
        result = check_ignore("cat /etc/passwd", PurePosixPath("/workspace"), ["**/.env"])
        assert result is not None
        assert "Access denied" in result


class TestSandboxShellToolConstruction:
    def test_auto_tempdir_created(self) -> None:
        shell = SandboxShellTool()
        assert shell.workdir.exists()
        assert shell.workdir.is_dir()

    def test_explicit_path_created(self, tmp_path: Path) -> None:
        target = tmp_path / "workspace"
        shell = SandboxShellTool(LocalEnvironment(target))
        assert shell.workdir == target
        assert target.exists()

    def test_workdir_is_readonly_property(self, tmp_path: Path) -> None:
        shell = SandboxShellTool(LocalEnvironment(tmp_path))
        with pytest.raises(AttributeError):
            shell.workdir = tmp_path  # type: ignore[misc]


class TestShellExecution:
    def _make_tool_call(self, command: str) -> ToolCallEvent:
        return ToolCallEvent(
            arguments=json.dumps({"command": command}),
            name="run_shell_command",
        )

    def _make_config(self, command: str, final_reply: str = "done") -> TestConfig:
        return TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent([self._make_tool_call(command)])),
            final_reply,
        )

    @pytest.mark.asyncio
    async def test_allowed_permits_matching_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = SandboxShellTool(LocalEnvironment(tmp_path), allowed=["echo"])
        agent = Agent("a", config=self._make_config(f"echo hello > {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "shell redirect bypass was not blocked"

    @pytest.mark.asyncio
    async def test_allowed_blocks_non_matching_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = SandboxShellTool(LocalEnvironment(tmp_path), allowed=["echo"])
        # "touch" is not in allowed — the file must NOT be created
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    @pytest.mark.asyncio
    async def test_blocked_rejects_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = SandboxShellTool(LocalEnvironment(tmp_path), blocked=["touch"])
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    @pytest.mark.asyncio
    async def test_env_merged_not_replaced(self, tmp_path: Path) -> None:
        # Use a helper script — avoids shell variable syntax differences
        # between bash ($VAR) and cmd.exe (%VAR%) across platforms.
        script = tmp_path / "check_env.py"
        script.write_text(
            "import os\n"
            "custom = os.environ.get('MY_CUSTOM_VAR', 'MISSING')\n"
            "path = os.environ.get('PATH', '')\n"
            "print(custom + '|' + path)\n"
        )
        shell = SandboxShellTool(LocalEnvironment(tmp_path, env_vars={"MY_CUSTOM_VAR": "hello"}))
        cmd = f'"{sys.executable}" check_env.py'
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        agent = Agent("a", config=self._make_config(cmd), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        result = tool_results[0]
        assert "hello|" in result, f"MY_CUSTOM_VAR not set: {result!r}"
        path_part = result.split("|", 1)[1] if "|" in result else ""
        assert path_part.strip(), f"PATH was lost — env was replaced instead of merged: {result!r}"

    @pytest.mark.asyncio
    async def test_timeout_returns_string_not_exception(self, tmp_path: Path) -> None:
        output_file = tmp_path / "timeout_result.txt"
        shell = SandboxShellTool(LocalEnvironment(tmp_path, timeout=1))
        # sleep 5 will time out after 1s
        cmd = f"sleep 5 && echo ok > {output_file}"
        agent = Agent("a", config=self._make_config(cmd), tools=[shell])
        # Must not raise — the tool should return a "timed out" string
        reply = await agent.ask("run it")
        assert await reply.content() == "done"
        # The file should not exist — command was killed
        assert not output_file.exists()

    @pytest.mark.asyncio
    async def test_ignore_blocks_env_file(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=password")
        shell = SandboxShellTool(LocalEnvironment(tmp_path), ignore=["**/.env"])

        tool_results: list[str] = []

        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        agent = Agent("a", config=self._make_config("cat .env"), tools=[shell])
        await agent.ask("show me .env", stream=stream)

        assert tool_results, "No tool result received"
        assert "Access denied" in tool_results[0], f"Expected 'Access denied' but got: {tool_results[0]!r}"
        assert "SECRET" not in tool_results[0], "File content leaked despite ignore pattern"

    @pytest.mark.asyncio
    async def test_exit_code_included_on_failure(self, tmp_path: Path) -> None:
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path))
        agent = Agent("a", config=self._make_config("exit 42"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 42" in tool_results[0], f"Exit code missing: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_exit_code_absent_on_success(self, tmp_path: Path) -> None:
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code" not in tool_results[0], f"Unexpected exit code in success: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_files_persist_between_ask_calls(self, tmp_path: Path) -> None:
        shell = SandboxShellTool(LocalEnvironment(tmp_path))
        agent = Agent(
            "a",
            config=TestConfig(
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[
                            ToolCallEvent(
                                arguments=json.dumps({"command": "echo 42 > counter.txt"}),
                                name="run_shell_command",
                            )
                        ]
                    )
                ),
                "created",
                ModelResponse(
                    tool_calls=ToolCallsEvent(
                        calls=[
                            ToolCallEvent(
                                arguments=json.dumps({"command": "cat counter.txt"}),
                                name="run_shell_command",
                            )
                        ]
                    )
                ),
                "read",
            ),
            tools=[shell],
        )

        reply1 = await agent.ask("create counter")
        assert (tmp_path / "counter.txt").exists()

        reply2 = await reply1.ask("read counter")
        assert await reply2.content() == "read"
        assert (tmp_path / "counter.txt").read_text().strip() == "42"

    @pytest.mark.asyncio
    async def test_output_truncated_when_exceeds_limit(self, tmp_path: Path) -> None:
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path, max_output=20))
        # Generate 100 chars of output
        agent = Agent("a", config=self._make_config("python3 -c \"print('x' * 100)\""), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        result = tool_results[0]
        assert "truncated" in result, f"Expected truncation note but got: {result!r}"
        # Output was 100 'x' chars; with max_output=20 only 20 should appear
        result = result.replace("TextInput", "")
        assert result.count("x") == 20, f"Expected exactly 20 'x' chars, got {result.count('x')}"

    @pytest.mark.asyncio
    async def test_output_not_truncated_within_limit(self, tmp_path: Path) -> None:
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path, max_output=1000))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "truncated" not in tool_results[0], "Unexpected truncation note for short output"

    @pytest.mark.asyncio
    async def test_timeout_returns_exit_code_124(self, tmp_path: Path) -> None:
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path, timeout=1))
        agent = Agent("a", config=self._make_config("sleep 5"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 124" in tool_results[0], f"Expected exit code 124 but got: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_readonly_blocks_write_commands(self, tmp_path: Path) -> None:
        output = tmp_path / "should_not_exist.txt"
        shell = SandboxShellTool(LocalEnvironment(tmp_path), readonly=True)
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was not blocked by readonly=True"

    @pytest.mark.asyncio
    async def test_readonly_allows_read_commands(self, tmp_path: Path) -> None:
        (tmp_path / "hello.txt").write_text("world")

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = SandboxShellTool(LocalEnvironment(tmp_path), readonly=True)
        agent = Agent("a", config=self._make_config("cat hello.txt"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "world" in tool_results[0], f"cat was blocked by readonly=True: {tool_results[0]!r}"

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="touch is POSIX-only")
    async def test_readonly_overridden_by_explicit_allowed(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = SandboxShellTool(
            LocalEnvironment(tmp_path),
            readonly=True,
            allowed=["touch"],  # user explicitly allows touch despite readonly
        )
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert output.exists(), "touch should be allowed when explicit allowed= overrides readonly"

    @pytest.mark.asyncio
    async def test_workdir_in_tool_description(self, tmp_path: Path) -> None:
        shell = SandboxShellTool(LocalEnvironment(tmp_path))

        schemas = await shell.schemas(None)  # type: ignore[arg-type]
        description = schemas[0].function.description
        assert str(tmp_path) in description, f"workdir not in description: {description!r}"


class _LoopRecordingSandbox:
    """Minimal Sandbox that records the event loop each exec runs on."""

    def __init__(self, loops: list[int]) -> None:
        self._loops = loops

    @property
    def workdir(self) -> PurePosixPath:
        return PurePosixPath("/workspace")

    @property
    def host_workdir(self) -> None:
        return None

    async def __aenter__(self) -> "_LoopRecordingSandbox":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def exec(
        self, argv: list[str], *, env: dict[str, str] | None = None, timeout: float | None = None
    ) -> ExecResult:
        self._loops.append(id(asyncio.get_running_loop()))
        return ExecResult(output="ok", exit_code=0)


class _RecordingFactory:
    """A SandboxFactory (not a SingletonFactory) yielding the recording sandbox."""

    def __init__(self) -> None:
        self.loops: list[int] = []
        self._sandbox = _LoopRecordingSandbox(self.loops)

    @asynccontextmanager
    async def open(self, context: object = None) -> "AsyncIterator[_LoopRecordingSandbox]":
        yield self._sandbox


class TestShellRunsInAgentLoop:
    """Regression for finding #1: SandboxShellTool's tool fn must be async so
    every command runs in the agent's single event loop — not via a throw-away
    asyncio.run() per call (which breaks loop-bound remote clients)."""

    def _tc(self, command: str) -> ToolCallEvent:
        return ToolCallEvent(arguments=json.dumps({"command": command}), name="run_shell_command")

    @pytest.mark.asyncio
    async def test_sequential_commands_share_one_event_loop(self) -> None:
        factory = _RecordingFactory()
        assert isinstance(factory, SandboxFactory)
        shell = SandboxShellTool(factory)
        config = TestConfig(
            ModelResponse(tool_calls=ToolCallsEvent([self._tc("echo a")])),
            ModelResponse(tool_calls=ToolCallsEvent([self._tc("echo b")])),
            "done",
        )
        agent = Agent("a", config=config, tools=[shell])

        await agent.ask("run two commands")

        assert len(factory.loops) == 2, f"expected 2 exec calls, got {factory.loops}"
        assert len(set(factory.loops)) == 1, (
            f"shell commands ran on different event loops — the sync asyncio.run regression is back: {factory.loops}"
        )
