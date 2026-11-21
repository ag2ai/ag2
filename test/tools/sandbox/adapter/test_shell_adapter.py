# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import locale
import os
import sys
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock

import pytest

from ag2 import Context
from ag2.tools.sandbox import SandboxFactory, WorkdirAware
from ag2.tools.sandbox.adapter import ShellAdapter
from ag2.tools.sandbox.filter import check_ignore
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

    @pytest.mark.parametrize(
        ("pattern", "command"),
        [
            pytest.param("/usr/bin/touch", "/usr/bin/touch pwned", id="absolute-path"),
            pytest.param("./danger.sh", "./danger.sh pwned", id="relative-path"),
            pytest.param("/usr/bin/touch pwned", "/usr/bin/touch  pwned", id="absolute-path-with-argument"),
            pytest.param("./danger.sh pwned", "'./danger.sh'  pwned", id="relative-path-with-argument"),
        ],
    )
    async def test_blocked_matches_path_based_prefixes(self, pattern: str, command: str) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, blocked=[pattern])
        result = await adapter.run(command)
        assert "Command not allowed" in result
        assert sandbox.execs == []

    async def test_blocked_path_prefix_allows_other_arguments(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, blocked=["./danger.sh pwned"])
        await adapter.run("./danger.sh safe")
        assert sandbox.execs == [["./danger.sh", "safe"]]

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

    @pytest.mark.parametrize("command", ["cat .en*", "cat .e?v", "cat .en[v]", "cat *"])
    async def test_ignore_denies_a_wildcard_that_matches_an_ignored_file(self, tmp_path: Path, command: str) -> None:
        # Some programs expand wildcards themselves (MSYS tools on Windows), so the
        # filter must not rely on the missing shell to keep the ignored file unread.
        (tmp_path / ".env").write_text("SECRET")
        adapter = ShellAdapter(LocalSandbox(tmp_path), ignore=[".env"])
        assert "Access denied" in await adapter.run(command)

    async def test_ignore_allows_a_wildcard_that_matches_nothing_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("fine")
        adapter = ShellAdapter(LocalSandbox(tmp_path), ignore=[".env"])
        assert "Access denied" not in await adapter.run("ls *.txt")

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("dd if=.env of=out.txt", id="dd-read"),
            pytest.param("dd if=a.txt of=.env", id="dd-write"),
            pytest.param("git diff --output=.env", id="git-output"),
            pytest.param("sort a.txt -o.env", id="short-option-attached"),
            pytest.param("sort a.txt -ro.env", id="short-options-attached"),
        ],
    )
    async def test_ignore_denies_path_embedded_in_a_word(self, command: str) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, ignore=[".env"])
        result = await adapter.run(command)
        assert "Access denied" in result
        assert sandbox.execs == []

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


SECRETS = ["**/.env", "*.key", "secrets/**"]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".env").write_text("OPENAI_API_KEY=SECRET-ENV\n")
    (tmp_path / "deploy.key").write_text("SECRET-KEY\n")
    (tmp_path / "secrets").mkdir()
    (tmp_path / "secrets" / "db.txt").write_text("DB_PASSWORD=SECRET-DB\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('hello')\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "app.py").write_text("print('hello docs')\n")
    return tmp_path


@pytest.mark.asyncio
class TestIgnoreWithDirectoryReaders:
    """``grep`` and ``diff`` open files inside a directory operand on their own."""

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param("grep -r SECRET", id="grep-recursive-no-path"),
            pytest.param("grep -r SECRET .", id="grep-recursive-dot"),
            pytest.param("grep -rn SECRET .", id="grep-bundled-options"),
            pytest.param("grep -R SECRET .", id="grep-dereference-recursive"),
            pytest.param("grep --recursive SECRET .", id="grep-long-option"),
            pytest.param("grep -d recurse SECRET", id="grep-directories-recurse"),
            pytest.param("grep --directories=recurse SECRET", id="grep-directories-recurse-long"),
            pytest.param("grep -r -m 5 SECRET", id="grep-option-with-value"),
            pytest.param("grep -r -e SECRET", id="grep-explicit-pattern"),
            pytest.param("grep -r src", id="grep-pattern-names-a-directory"),
            pytest.param("grep -r . secrets", id="grep-ignored-directory-contents"),
            pytest.param("egrep -r SECRET .", id="egrep"),
            pytest.param("fgrep -r SECRET .", id="fgrep"),
            pytest.param("diff -rN src .", id="diff-recursive"),
            pytest.param("diff -N . src", id="diff-directories-without-recursion"),
            pytest.param("diff --new-file secrets src", id="diff-ignored-directory"),
        ],
    )
    async def test_readonly_refuses_reading_ignored_files_through_a_directory(
        self, project: Path, command: str
    ) -> None:
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)

        result = await adapter.run(command)

        assert "Access denied" in result
        assert "SECRET" not in result

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            pytest.param("grep -r hello src", "hello", id="grep-recursive-clean-directory"),
            pytest.param("grep -rn hello src docs", "hello docs", id="grep-several-clean-directories"),
            pytest.param("grep hello src/app.py", "hello", id="grep-file"),
            pytest.param("diff -r src docs", "hello docs", id="diff-clean-directories"),
        ],
    )
    async def test_readonly_still_reads_directories_without_ignored_files(
        self, project: Path, command: str, expected: str
    ) -> None:
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)

        result = await adapter.run(command)

        assert expected in result
        assert "Access denied" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
    async def test_symlink_inside_a_directory_counts_as_its_target(self, project: Path) -> None:
        (project / "src" / "settings").symlink_to(project / ".env")
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)

        result = await adapter.run("grep -R SECRET src")

        assert "Access denied" in result
        assert "SECRET" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
    async def test_symlink_out_of_the_workdir_inside_a_directory_is_refused(
        self, project: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        outside = tmp_path_factory.mktemp("outside")
        (outside / "id_rsa").write_text("SECRET-OUTSIDE\n")
        (project / "src" / "home").symlink_to(outside)
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)

        result = await adapter.run("grep -R SECRET src")

        assert "Access denied" in result
        assert "SECRET" not in result

    async def test_remote_backend_refuses_directory_readers(self) -> None:
        # The contents of a remote directory cannot be checked from the host.
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, readonly=True, ignore=SECRETS)

        result = await adapter.run("grep -r SECRET .")

        assert "Access denied" in result
        assert sandbox.execs == []

    async def test_remote_backend_without_ignore_runs_directory_readers(self) -> None:
        sandbox = RecordingSandbox()
        adapter = ShellAdapter(sandbox, readonly=True)

        await adapter.run("grep -r hello .")

        assert sandbox.execs == [["grep", "-r", "hello", "."]]

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
    @pytest.mark.parametrize("operand", ["src", "src/conf"])
    async def test_rule_scoped_to_a_symlinked_directory_applies_to_its_contents(
        self, project: Path, operand: str
    ) -> None:
        (project / "config").mkdir()
        (project / "config" / "prod.txt").write_text("SECRET-CONF\n")
        (project / "src" / "conf").symlink_to(project / "config")
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=["src/conf/**"])

        result = await adapter.run(f"grep -R SECRET-CONF {operand}")

        assert "Access denied" in result
        assert "SECRET" not in result

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra privileges on Windows")
    async def test_symlink_loop_in_a_clean_directory_is_allowed(self, project: Path) -> None:
        (project / "src" / "loop").symlink_to(project / "src")

        assert check_ignore("grep -R hello src", project, SECRETS) is None

    @pytest.mark.skipif(sys.platform == "win32", reason="directory permissions differ on Windows")
    @pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="root can list any directory")
    async def test_directory_that_cannot_be_listed_is_refused(self, project: Path) -> None:
        locked = project / "src" / "locked"
        locked.mkdir()
        locked.chmod(0)
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)
        try:
            result = await adapter.run("grep -r hello src")
        finally:
            locked.chmod(0o755)

        assert "Access denied" in result

    @pytest.mark.parametrize("command", ["grep -r SECRET secret*", "diff -N secret? docs"])
    async def test_wildcard_naming_a_directory_with_ignored_files_is_refused(self, project: Path, command: str) -> None:
        # Some programs expand wildcards themselves (MSYS tools on Windows).
        adapter = ShellAdapter(LocalSandbox(project), readonly=True, ignore=SECRETS)

        result = await adapter.run(command)

        assert "Access denied" in result

    async def test_directory_larger_than_the_walk_budget_is_refused(self, project: Path) -> None:
        for i in range(10):
            (project / "src" / f"module_{i}.py").write_text("print('hello')\n")

        assert check_ignore("grep -r hello src", project, SECRETS, max_entries=5) is not None
        assert check_ignore("grep -r hello src", project, SECRETS, max_entries=100) is None


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

    async def test_run_keeps_undecodable_output_and_exit_code(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = ShellAdapter(sandbox)
        result = await adapter.run(r"printf 'error: \201' >&2; exit 7")
        expected = b"error: \x81".decode(locale.getpreferredencoding(False), errors="replace")
        assert result == f"{expected}\n[exit code: 7]"


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
