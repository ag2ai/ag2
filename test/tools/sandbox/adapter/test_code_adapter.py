# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import sys
from pathlib import Path

import pytest

from ag2.tools.sandbox import CodeAdapter, LanguageRunner
from ag2.tools.sandbox.local import LocalSandbox


@pytest.mark.asyncio
class TestCodeAdapterInline:
    async def test_python_runs_inline(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(sandbox, languages=("python",))
        result = await adapter.run("print(40+2)", "python")
        assert result.exit_code == 0
        assert "42" in result.output

    @pytest.mark.skipif(sys.platform == "win32", reason="bash is POSIX-only")
    async def test_bash_runs_inline(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(sandbox, languages=("bash",))
        result = await adapter.run("echo hello", "bash")
        assert result.exit_code == 0
        assert "hello" in result.output


@pytest.mark.asyncio
class TestCodeAdapterFileMode:
    async def test_javascript_via_put_file_when_node_available(self, tmp_path: Path) -> None:
        if shutil.which("node") is None:
            pytest.skip("node not installed")
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(sandbox, languages=("python", "javascript"))
        result = await adapter.run("console.log(1+1)", "javascript")
        assert result.exit_code == 0
        assert "2" in result.output

    async def test_file_mode_cleans_up_temp_script(self, tmp_path: Path) -> None:
        if shutil.which("node") is None:
            pytest.skip("node not installed")
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(sandbox, languages=("python", "javascript"))
        await adapter.run("console.log(1)", "javascript")
        # The temp ag2_*.js script must not linger in a persistent workdir.
        assert list(tmp_path.glob("ag2_*.js")) == []


@pytest.mark.asyncio
class TestCodeAdapterLanguages:
    async def test_disabled_language_returns_error(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(sandbox, languages=("python",))
        result = await adapter.run("echo hi", "bash")
        assert result.exit_code == 2
        assert "not enabled" in result.output

    async def test_custom_runner_extends_languages(self, tmp_path: Path) -> None:
        sandbox = LocalSandbox(tmp_path)
        adapter = CodeAdapter(
            sandbox,
            languages=("python",),
            runners={"python": LanguageRunner(inline_argv=("python3", "-c"))},
        )
        result = await adapter.run("print('via-python3')", "python")
        assert result.exit_code == 0
        assert "via-python3" in result.output


def test_unknown_language_raises_on_construction(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(ValueError, match="LanguageRunner"):
        CodeAdapter(sandbox, languages=("ocaml",))  # type: ignore[arg-type]


def test_supported_languages_property(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    adapter = CodeAdapter(sandbox, languages=("python", "bash"))
    assert adapter.supported_languages == ("python", "bash")


class TestLanguageRunnerValidation:
    def test_inline_only_is_valid(self) -> None:
        LanguageRunner(inline_argv=("python", "-c"))

    def test_file_only_is_valid(self) -> None:
        LanguageRunner(file_extension="js", file_runner_argv=("node",))

    def test_mixed_mode_rejected(self) -> None:
        with pytest.raises(ValueError):
            LanguageRunner(inline_argv=("python", "-c"), file_extension="py")

    def test_empty_runner_rejected(self) -> None:
        with pytest.raises(ValueError):
            LanguageRunner()

    def test_partial_file_runner_rejected(self) -> None:
        with pytest.raises(ValueError):
            LanguageRunner(file_extension="js")
