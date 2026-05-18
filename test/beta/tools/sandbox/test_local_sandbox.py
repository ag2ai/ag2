# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from autogen.beta.tools.sandbox import ExecResult, LocalSandbox, Sandbox


@pytest.mark.asyncio
class TestLocalSandboxExec:
    async def test_argv_form_runs_command(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["python3", "-c", "print('hi')"])
        assert result == ExecResult(output="hi", exit_code=0)

    async def test_shell_mode_runs_pipeline(self) -> None:
        # No spaces around `&&`: cmd.exe preserves trailing whitespace as
        # part of echo's argument, bash trims either way.  Tight form gives
        # identical output on both shells.
        sandbox = LocalSandbox()
        result = await sandbox.exec(["echo hello&&echo world"], shell=True)
        assert result == ExecResult(output="hello\nworld", exit_code=0)

    async def test_workdir_is_used_as_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "marker.txt").write_text("hello")
        sandbox = LocalSandbox(tmp_path)
        result = await sandbox.exec(["cat", "marker.txt"])
        assert result == ExecResult(output="hello", exit_code=0)

    async def test_extra_env_vars_are_merged(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["sh", "-c", "echo $FOO"], env={"FOO": "bar"})
        assert result == ExecResult(output="bar", exit_code=0)

    async def test_nonzero_exit_code_preserved(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["sh", "-c", "exit 7"])
        assert result.exit_code == 7

    async def test_default_timeout_returns_124(self) -> None:
        sandbox = LocalSandbox(timeout=0.1)
        result = await sandbox.exec(["sleep", "5"])
        assert result.exit_code == 124
        assert "timed out" in result.output.lower()

    async def test_per_call_timeout_overrides_default(self) -> None:
        sandbox = LocalSandbox(timeout=10)
        result = await sandbox.exec(["sleep", "5"], timeout=0.1)
        assert result.exit_code == 124

    async def test_output_is_truncated_when_too_long(self) -> None:
        sandbox = LocalSandbox(max_output=10)
        long = "x" * 200
        result = await sandbox.exec(["python3", "-c", f"print('{long}')"])
        assert result.exit_code == 0
        assert "[truncated:" in result.output

    async def test_command_not_found_returns_127(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec(["this-command-does-not-exist-xyz"])
        assert result.exit_code == 127

    async def test_empty_argv_returns_failure(self) -> None:
        sandbox = LocalSandbox()
        result = await sandbox.exec([])
        assert result == ExecResult(output="", exit_code=2)

    async def test_aclose_blocks_further_exec(self) -> None:
        sandbox = LocalSandbox()
        await sandbox.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            await sandbox.exec(["echo", "hi"])

    async def test_aclose_is_idempotent(self) -> None:
        sandbox = LocalSandbox()
        await sandbox.aclose()
        await sandbox.aclose()


@pytest.mark.asyncio
class TestLocalSandboxProperties:
    async def test_workdir_returns_user_path(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        assert sandbox.workdir == tmp_path.resolve()

    async def test_workdir_temp_when_path_none(self) -> None:
        sandbox = LocalSandbox()
        assert sandbox.workdir.exists()
        assert "ag2_sandbox_" in sandbox.workdir.name

    async def test_workdir_is_created_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir"
        sandbox = LocalSandbox(nested)
        assert sandbox.workdir.exists()
        assert sandbox.workdir == nested.resolve()

    async def test_supported_languages_includes_python_by_default(self) -> None:
        sandbox = LocalSandbox()
        assert "python" in sandbox.supported_languages

    async def test_supported_languages_can_be_overridden(self) -> None:
        sandbox = LocalSandbox(languages=("python",))
        assert sandbox.supported_languages == ("python",)

    async def test_satisfies_sandbox_protocol(self) -> None:
        sandbox = LocalSandbox()
        assert isinstance(sandbox, Sandbox)
