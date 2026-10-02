# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest

from ag2 import Context
from ag2.tools.sandbox import SandboxFactory, WorkdirAware
from ag2.tools.sandbox.adapter import ShellAdapter
from ag2.tools.sandbox.local import LocalSandbox
from test.tools.sandbox._helpers import RecordingFactory, RecordingSandbox, WorkdirDeclaringFactory


@pytest.mark.asyncio
class TestShellAdapterFiltering:
    async def test_allowed_blocks_non_matching_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, allowed=["echo"])
        result = await adapter.run("touch file.txt")
        assert "Command not allowed" in result

    async def test_blocked_rejects_matching_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, blocked=["rm -rf"])
        result = await adapter.run("rm -rf /workspace")
        assert "Command not allowed" in result

    async def test_ignore_denies_access_to_matching_path(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("SECRET=1")
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat .env")
        assert "Access denied" in result

    async def test_ignore_applies_on_remote_backend(self) -> None:
        # A remote backend has no host workdir; ignore must still apply by
        # matching literal argv paths against the sandbox-side workdir.
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat .env")
        assert "Access denied" in result
        assert sandbox.execs == []  # blocked before reaching the backend

    async def test_ignore_allows_non_matching_on_remote_backend(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, ignore=["**/.env"])
        result = await adapter.run("cat README.md")
        assert "ok" in result
        assert len(sandbox.execs) == 1

    async def test_readonly_blocks_writes_by_default(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, readonly=True)
        result = await adapter.run("touch new.txt")
        assert "Command not allowed" in result

    async def test_readonly_allows_read_commands(self, tmp_path: Path) -> None:
        # Not ``echo``: restricted mode runs it from PATH, and nlip-server ships a broken ``echo`` into the venv.
        (tmp_path / "hello.txt").write_text("hello")
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox, readonly=True)
        result = await adapter.run("cat hello.txt")
        assert "hello" in result

    @pytest.mark.parametrize(
        ("allowed", "readonly"),
        [
            pytest.param(None, True, id="readonly"),
            pytest.param(["git"], False, id="allowed"),
        ],
    )
    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("git status --short\ntouch pwned", id="newline"),
            pytest.param("git status --short\rtouch pwned", id="carriage-return"),
            pytest.param("git status --short & touch pwned", id="background"),
            pytest.param("git diff <(touch pwned)", id="process-substitution"),
        ],
    )
    async def test_restricted_mode_blocks_second_command(
        self, tmp_path: Path, allowed: list[str] | None, readonly: bool, command: str
    ) -> None:
        adapter = ShellAdapter(LocalSandbox(tmp_path), allowed=allowed, readonly=readonly)
        result = await adapter.run(command)
        assert "Command not allowed" in result
        assert not (tmp_path / "pwned").exists()

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("env touch pwned", id="env"),
            pytest.param("find . -exec touch pwned {} +", id="find"),
            pytest.param("sort -o pwned /dev/null", id="sort"),
            pytest.param("uniq /dev/null pwned", id="uniq"),
            pytest.param("file -C -m pwned", id="file"),
            pytest.param("git diff --no-index --output=pwned /dev/null /dev/null", id="git"),
        ],
    )
    async def test_readonly_leaves_out_commands_that_can_write_or_execute(self, tmp_path: Path, command: str) -> None:
        adapter = ShellAdapter(LocalSandbox(tmp_path), readonly=True)
        result = await adapter.run(command)
        assert "Command not allowed" in result
        assert not list(tmp_path.glob("pwned*"))

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("find . -maxdepth 0 {-exec,} touch pwned {} +", id="brace"),
            pytest.param("find . -maxdepth 0 -e${EMPTY}xec touch pwned {} +", id="variable"),
        ],
    )
    async def test_restricted_mode_does_not_expand_arguments(self, tmp_path: Path, command: str) -> None:
        adapter = ShellAdapter(LocalSandbox(tmp_path), allowed=["find"])
        await adapter.run(command)
        assert not (tmp_path / "pwned").exists()

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("echo x; touch pwned", id="semicolon"),
            pytest.param("true && touch pwned", id="and"),
            pytest.param("echo $(touch pwned)", id="substitution"),
            pytest.param("echo `touch pwned`", id="backtick"),
            pytest.param("echo x | touch pwned", id="pipe"),
            pytest.param("echo x\ntouch pwned", id="newline"),
        ],
    )
    async def test_blocked_alone_does_not_run_chained_commands(self, tmp_path: Path, command: str) -> None:
        adapter = ShellAdapter(LocalSandbox(tmp_path), blocked=["touch"])
        result = await adapter.run(command)
        assert "Command not allowed" in result
        assert not (tmp_path / "pwned").exists()

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("touch pwned", id="plain"),
            pytest.param("touch  pwned", id="double-space"),
            pytest.param("/usr/bin/touch pwned", id="absolute-path"),
            pytest.param("'touch' pwned", id="quoted"),
        ],
    )
    async def test_blocked_matches_the_argv(self, tmp_path: Path, command: str) -> None:
        adapter = ShellAdapter(LocalSandbox(tmp_path), blocked=["touch"])
        result = await adapter.run(command)
        assert "Command not allowed" in result
        assert not (tmp_path / "pwned").exists()

    async def test_blocked_applies_on_top_of_allowed(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, allowed=["git"], blocked=["git push"])
        result = await adapter.run("git  push")
        assert "Command not allowed" in result
        assert sandbox.execs == []

    async def test_blocked_runs_other_commands_as_argv(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, blocked=["rm"])
        await adapter.run("echo 'a; b' \"c | d\"")
        assert sandbox.execs == [["echo", "a; b", "c | d"]]

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("cat .e$(echo)nv", id="substitution"),
            pytest.param("cat .en*", id="glob"),
            pytest.param("e=.env; cat $e", id="variable"),
        ],
    )
    async def test_ignore_alone_does_not_leak_through_shell_expansion(self, tmp_path: Path, command: str) -> None:
        (tmp_path / ".env").write_text("SECRET")
        adapter = ShellAdapter(LocalSandbox(tmp_path), ignore=[".env"])
        result = await adapter.run(command)
        assert "SECRET" not in result

    async def test_blocked_or_ignore_alone_switches_on_restricted_mode(self) -> None:
        assert ShellAdapter(RecordingSandbox(), blocked=["rm"]).restricted
        assert ShellAdapter(RecordingSandbox(), ignore=[".env"]).restricted
        assert not ShellAdapter(RecordingSandbox()).restricted

    async def test_restricted_mode_does_not_expand_globs(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, readonly=True)
        await adapter.run("ls *.py")
        assert sandbox.execs == [["ls", "*.py"]]

    async def test_allowed_matches_whole_words_of_argv(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, allowed=["git log"])
        result = await adapter.run("git -c core.pager=touch log")
        assert "Command not allowed" in result
        assert sandbox.execs == []

    async def test_restricted_mode_rejects_unbalanced_quotes(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, readonly=True)
        result = await adapter.run("cat 'README.md")
        assert "unbalanced quotes" in result
        assert sandbox.execs == []

    async def test_unrestricted_mode_runs_through_shell(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox)
        await adapter.run("echo a | cat")
        assert sandbox.execs == [["sh", "-c", "echo a | cat"]]


@pytest.mark.asyncio
class TestShellAdapterAsync:
    async def test_run_executes_command(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox)
        result = await adapter.run("echo hi")
        assert "hi" in result

    async def test_run_includes_exit_code_on_failure(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox)
        result = await adapter.run("exit 7")
        assert "exit code: 7" in result


@pytest.mark.asyncio
class TestShellAdapterWithFactory:
    async def test_factory_opens_per_call(self, tmp_path: Path) -> None:
        factory = RecordingFactory(LocalSandbox(tmp_path))
        adapter = ShellAdapter(factory)

        await adapter.run("echo a")
        await adapter.run("echo b")

        assert len(factory.contexts) == 2

    async def test_context_variables_forwarded_to_factory(self, tmp_path: Path) -> None:
        factory = RecordingFactory(LocalSandbox(tmp_path))
        adapter = ShellAdapter(factory)
        ctx = Context(stream=MagicMock(), variables={"x": "value"})

        await adapter.run("echo a", context=ctx)

        assert factory.contexts == [ctx]


class TestShellAdapterWorkdir:
    """A remote factory is not bound to a sandbox until it is opened, so the
    workdir reported up front is whatever the factory itself declares.
    """

    def test_undeclared_factory_reports_conventional_workspace(self, tmp_path: Path) -> None:
        adapter = ShellAdapter(RecordingFactory(LocalSandbox(tmp_path)))

        assert adapter.workdir == PurePosixPath("/workspace")

    def test_workdir_aware_factory_is_reported(self) -> None:
        factory = WorkdirDeclaringFactory(PurePosixPath("/home/agent"))

        assert isinstance(factory, WorkdirAware)
        assert ShellAdapter(factory).workdir == PurePosixPath("/home/agent")

    def test_declaring_a_workdir_is_optional_for_a_factory(self, tmp_path: Path) -> None:
        # WorkdirAware must stay separate from SandboxFactory: every backend that
        # predates it still satisfies the factory protocol without declaring one.
        factory = RecordingFactory(LocalSandbox(tmp_path))

        assert isinstance(factory, SandboxFactory)
        assert not isinstance(factory, WorkdirAware)
