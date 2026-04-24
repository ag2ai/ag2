# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.events import ToolCallEvent, ToolCallsEvent, ToolResultEvent
from autogen.beta.events.types import ModelResponse
from autogen.beta.testing import TestConfig
from autogen.beta.tools import LocalShellTool
from autogen.beta.tools.shell import LocalShellEnvironment
from autogen.beta.tools.shell.environment.base import check_ignore, matches


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


class TestLocalShellToolConstruction:
    def test_auto_tempdir_created(self) -> None:
        shell = LocalShellTool()
        assert shell.workdir.exists()
        assert shell.workdir.is_dir()

    def test_explicit_path_created(self, tmp_path: Path) -> None:
        target = tmp_path / "workspace"
        shell = LocalShellTool(environment=target)
        assert shell.workdir == target
        assert target.exists()

    def test_workdir_is_readonly_property(self, tmp_path: Path) -> None:
        shell = LocalShellTool(environment=tmp_path)
        with pytest.raises(AttributeError):
            shell.workdir = tmp_path  # type: ignore[misc]


class TestShellExecution:
    """These tests call the tool function directly via the agent + TestConfig."""

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
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, allowed=["echo"]))
        agent = Agent("a", config=self._make_config(f"echo hello > {output}"), tools=[shell])
        await agent.ask("run it")
        # Command was allowed — file must exist with expected content
        assert output.exists(), "echo was allowed but file was not created"
        assert output.read_text().strip() == "hello"

    @pytest.mark.asyncio
    async def test_allowed_blocks_non_matching_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, allowed=["echo"]))
        # "touch" is not in allowed — the file must NOT be created
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    # ── blocked ───────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_blocked_rejects_command(self, tmp_path: Path) -> None:
        output = tmp_path / "out.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, blocked=["touch"]))
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was blocked but file was created anyway"

    # ── env merging ───────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_env_merged_not_replaced(self, tmp_path: Path) -> None:
        """Extra env vars must be added on top of os.environ, not replace it."""
        # Write a helper script — avoids shell variable syntax differences
        # between bash ($VAR) and cmd.exe (%VAR%) across platforms.
        script = tmp_path / "check_env.py"
        script.write_text(
            "import os\n"
            "custom = os.environ.get('MY_CUSTOM_VAR', 'MISSING')\n"
            "path = os.environ.get('PATH', '')\n"
            "print(custom + '|' + path)\n"
        )
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                env={"MY_CUSTOM_VAR": "hello"},
            )
        )
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

    # ── timeout ───────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_returns_string_not_exception(self, tmp_path: Path) -> None:
        """A timed-out command must return an error string, not raise."""
        output_file = tmp_path / "timeout_result.txt"
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                timeout=1,
            )
        )
        # sleep 5 will time out after 1s
        cmd = f"sleep 5 && echo ok > {output_file}"
        agent = Agent("a", config=self._make_config(cmd), tools=[shell])
        # Must not raise — the tool should return a "timed out" string
        reply = await agent.ask("run it")
        assert await reply.content() == "done"
        # The file should not exist — command was killed
        assert not output_file.exists()

    # ── ignore ────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_ignore_blocks_env_file(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=password")
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, ignore=["**/.env"]))

        tool_results: list[str] = []

        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        agent = Agent("a", config=self._make_config("cat .env"), tools=[shell])
        await agent.ask("show me .env", stream=stream)

        assert tool_results, "No tool result received"
        assert "Access denied" in tool_results[0], f"Expected 'Access denied' but got: {tool_results[0]!r}"
        assert "SECRET" not in tool_results[0], "File content leaked despite ignore pattern"

    # ── exit code ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_exit_code_included_on_failure(self, tmp_path: Path) -> None:
        """A failed command must include [exit code: N] in the tool result."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
        agent = Agent("a", config=self._make_config("exit 42"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 42" in tool_results[0], f"Exit code missing: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_exit_code_absent_on_success(self, tmp_path: Path) -> None:
        """A successful command must NOT include [exit code: ...] in the result."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code" not in tool_results[0], f"Unexpected exit code in success: {tool_results[0]!r}"

    # ── filesystem persistence ────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_files_persist_between_ask_calls(self, tmp_path: Path) -> None:
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))
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

    # ── output truncation ─────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_output_truncated_when_exceeds_limit(self, tmp_path: Path) -> None:
        """Output longer than max_output must be cut with a note appended."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, max_output=20))
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
        """Short output must be returned as-is without any truncation note."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, max_output=1000))
        agent = Agent("a", config=self._make_config("echo hello"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "truncated" not in tool_results[0], "Unexpected truncation note for short output"

    # ── timeout exit code 124 ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_returns_exit_code_124(self, tmp_path: Path) -> None:
        """Timed-out commands must include [exit code: 124] (Unix convention)."""
        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, timeout=1))
        agent = Agent("a", config=self._make_config("sleep 5"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "exit code: 124" in tool_results[0], f"Expected exit code 124 but got: {tool_results[0]!r}"

    # ── readonly ──────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_readonly_blocks_write_commands(self, tmp_path: Path) -> None:
        """readonly=True must block touch, rm, mkdir."""
        output = tmp_path / "should_not_exist.txt"
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, readonly=True))
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert not output.exists(), "touch was not blocked by readonly=True"

    @pytest.mark.asyncio
    async def test_readonly_allows_read_commands(self, tmp_path: Path) -> None:
        """readonly=True must allow cat, ls, grep."""
        (tmp_path / "hello.txt").write_text("world")

        tool_results: list[str] = []
        stream = MemoryStream()
        stream.where(ToolResultEvent).subscribe(lambda e: tool_results.append(str(e.result)))

        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path, readonly=True))
        agent = Agent("a", config=self._make_config("cat hello.txt"), tools=[shell])
        await agent.ask("run it", stream=stream)

        assert tool_results, "No tool result received"
        assert "world" in tool_results[0], f"cat was blocked by readonly=True: {tool_results[0]!r}"

    @pytest.mark.asyncio
    async def test_readonly_overridden_by_explicit_allowed(self, tmp_path: Path) -> None:
        """explicit allowed= takes precedence over readonly=True."""
        output = tmp_path / "out.txt"
        shell = LocalShellTool(
            environment=LocalShellEnvironment(
                path=tmp_path,
                readonly=True,
                allowed=["touch"],  # user explicitly allows touch despite readonly
            )
        )
        agent = Agent("a", config=self._make_config(f"touch {output}"), tools=[shell])
        await agent.ask("run it")
        assert output.exists(), "touch should be allowed when explicit allowed= overrides readonly"

    # ── workdir in tool description ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_workdir_in_tool_description(self, tmp_path: Path) -> None:
        """The shell tool description must include the working directory path."""
        shell = LocalShellTool(environment=LocalShellEnvironment(path=tmp_path))

        schemas = await shell.schemas(None)  # type: ignore[arg-type]
        description = schemas[0].function.description
        assert str(tmp_path) in description, f"workdir not in description: {description!r}"
