# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for DaytonaCodeExecutor — requires a live Daytona instance."""

import os
from pathlib import Path

import pytest

from autogen.coding import CodeBlock

try:
    import dotenv

    from autogen.coding import DaytonaCodeExecutor, DaytonaCodeResult

    _has_daytona = DaytonaCodeExecutor is not None
except ImportError:
    _has_daytona = False

pytestmark = pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def load_env():
    """Load .env from repo root if present."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)


@pytest.fixture(scope="module")
def executor():
    """A single sandbox shared across all integration tests in this module."""
    if not os.getenv("DAYTONA_API_KEY"):
        pytest.skip("DAYTONA_API_KEY not set — skipping integration tests")
    with DaytonaCodeExecutor(timeout=60) as ex:
        yield ex


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaPythonIntegration:
    def test_basic_print(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="print('hello from daytona')", language="python")])
        assert result.exit_code == 0, result.output
        assert "hello from daytona" in result.output

    def test_math_computation(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="import math\nprint(math.sqrt(144))", language="python")])
        assert result.exit_code == 0, result.output
        assert "12.0" in result.output

    def test_multiline_code(self, executor):
        code = "\n".join([
            "total = 0",
            "for i in range(1, 6):",
            "    total += i",
            "print(total)",
        ])
        result = executor.execute_code_blocks([CodeBlock(code=code, language="python")])
        assert result.exit_code == 0, result.output
        assert "15" in result.output

    def test_runtime_error_returns_nonzero_exit(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="print(undefined_variable)", language="python")])
        assert result.exit_code != 0
        assert "NameError" in result.output or "undefined_variable" in result.output

    def test_syntax_error_returns_nonzero_exit(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="def broken(:\n    pass", language="python")])
        assert result.exit_code != 0

    def test_py_alias_works(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="print('alias ok')", language="py")])
        assert result.exit_code == 0, result.output
        assert "alias ok" in result.output


# ---------------------------------------------------------------------------
# Bash / sh
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaBashIntegration:
    def test_basic_echo(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="echo 'bash works'", language="bash")])
        assert result.exit_code == 0, result.output
        assert "bash works" in result.output

    def test_multiline_bash(self, executor):
        code = "for i in 1 2 3; do\n  echo $i\ndone"
        result = executor.execute_code_blocks([CodeBlock(code=code, language="bash")])
        assert result.exit_code == 0, result.output
        assert "1" in result.output
        assert "3" in result.output

    def test_sh_alias(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="echo 'sh works'", language="sh")])
        assert result.exit_code == 0, result.output
        assert "sh works" in result.output

    def test_bash_exit_code_propagated(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="exit 42", language="bash")])
        assert result.exit_code == 42


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaJavaScriptIntegration:
    def test_basic_console_log(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="console.log('js works')", language="javascript")])
        assert result.exit_code == 0, result.output
        assert "js works" in result.output

    def test_js_alias(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="console.log('alias ok')", language="js")])
        assert result.exit_code == 0, result.output
        assert "alias ok" in result.output

    def test_js_arithmetic(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="console.log(6 * 7)", language="javascript")])
        assert result.exit_code == 0, result.output
        assert "42" in result.output


# ---------------------------------------------------------------------------
# TypeScript
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaTypeScriptIntegration:
    def test_basic_ts(self, executor):
        code = "const msg: string = 'ts works';\nconsole.log(msg);"
        result = executor.execute_code_blocks([CodeBlock(code=code, language="typescript")])
        assert result.exit_code == 0, result.output
        assert "ts works" in result.output

    def test_ts_alias(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="const x: number = 42;\nconsole.log(x);", language="ts")])
        assert result.exit_code == 0, result.output
        assert "42" in result.output

    def test_ts_with_import_statement(self, executor):
        """Verify the CommonJS flags prevent 'Cannot use import statement outside a module'."""
        code = "import * as path from 'path';\nconsole.log(path.join('a', 'b'));"
        result = executor.execute_code_blocks([CodeBlock(code=code, language="typescript")])
        assert result.exit_code == 0, result.output
        assert "a/b" in result.output or "a\\b" in result.output


# ---------------------------------------------------------------------------
# Multiple blocks
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaMultipleBlocksIntegration:
    def test_multiple_blocks_outputs_joined(self, executor):
        result = executor.execute_code_blocks([
            CodeBlock(code="print('block one')", language="python"),
            CodeBlock(code="print('block two')", language="python"),
        ])
        assert result.exit_code == 0, result.output
        assert "block one" in result.output
        assert "block two" in result.output

    def test_multiple_blocks_stops_on_first_failure(self, executor):
        result = executor.execute_code_blocks([
            CodeBlock(code="print(undefined_variable)", language="python"),
            CodeBlock(code="print('should not appear')", language="python"),
        ])
        assert result.exit_code != 0
        assert "should not appear" not in result.output

    def test_mixed_languages(self, executor):
        result = executor.execute_code_blocks([
            CodeBlock(code="print('from python')", language="python"),
            CodeBlock(code="echo 'from bash'", language="bash"),
        ])
        assert result.exit_code == 0, result.output
        assert "from python" in result.output
        assert "from bash" in result.output


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaResultMetadata:
    def test_sandbox_id_is_present(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="print('hi')", language="python")])
        assert result.sandbox_id is not None
        assert len(result.sandbox_id) > 0

    def test_result_is_daytona_code_result(self, executor):
        result = executor.execute_code_blocks([CodeBlock(code="x=1", language="python")])
        assert isinstance(result, DaytonaCodeResult)

    def test_sandbox_reused_across_calls(self, executor):
        """All calls on the same executor must share a single sandbox."""
        result1 = executor.execute_code_blocks([CodeBlock(code="print('first')", language="python")])
        result2 = executor.execute_code_blocks([CodeBlock(code="print('second')", language="python")])
        assert result1.sandbox_id == result2.sandbox_id

    def test_empty_code_blocks_returns_success(self, executor):
        result = executor.execute_code_blocks([])
        assert result.exit_code == 0
        assert result.output == ""


# ---------------------------------------------------------------------------
# Custom image — string and declarative Image object
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaCustomImageIntegration:
    def test_image_string(self):
        """Sandbox created from a plain image name string executes code correctly."""
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        with DaytonaCodeExecutor(timeout=60, image="python:3.12-slim") as executor:
            result = executor.execute_code_blocks([CodeBlock(code="print('image string ok')", language="python")])
        assert result.exit_code == 0, result.output
        assert "image string ok" in result.output

    def test_image_object(self):
        """Sandbox created from a declarative Image object executes code correctly."""
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        from daytona import Image

        declarative_image = Image.base("python:3.12-slim")
        with DaytonaCodeExecutor(timeout=60, image=declarative_image) as executor:
            result = executor.execute_code_blocks([CodeBlock(code="print('image object ok')", language="python")])
        assert result.exit_code == 0, result.output
        assert "image object ok" in result.output

    def test_image_object_with_pip_install(self):
        """Declarative image with pip_install makes packages available in sandbox."""
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        from daytona import Image

        declarative_image = Image.base("python:3.12-slim").pip_install("requests")
        with DaytonaCodeExecutor(timeout=120, image=declarative_image) as executor:
            result = executor.execute_code_blocks([
                CodeBlock(code="import requests\nprint(requests.__version__)", language="python")
            ])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# env_vars
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaEnvVarsIntegration:
    def test_env_vars_available_in_sandbox(self):
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        with DaytonaCodeExecutor(timeout=60, env_vars={"MY_SECRET": "hunter2"}) as executor:
            result = executor.execute_code_blocks([
                CodeBlock(code="import os\nprint(os.environ.get('MY_SECRET', 'not found'))", language="python")
            ])
        assert result.exit_code == 0, result.output
        assert "hunter2" in result.output


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaRestartIntegration:
    def test_restart_creates_fresh_sandbox(self):
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        with DaytonaCodeExecutor(timeout=60) as executor:
            original_id = executor._sandbox.id
            executor.restart()
            assert executor._sandbox.id != original_id


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaContextManagerIntegration:
    def test_context_manager_cleans_up(self):
        if not os.getenv("DAYTONA_API_KEY"):
            pytest.skip("DAYTONA_API_KEY not set")
        with DaytonaCodeExecutor(timeout=60) as executor:
            result = executor.execute_code_blocks([CodeBlock(code="print('ok')", language="python")])
            assert result.exit_code == 0, result.output
        # After __exit__, sandbox should be None
        assert executor._sandbox is None
