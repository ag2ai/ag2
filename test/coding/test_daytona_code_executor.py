# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DaytonaCodeExecutor."""

from unittest.mock import Mock, patch

import pytest

from autogen.coding import CodeBlock, MarkdownCodeExtractor

try:
    from daytona import DaytonaError, DaytonaRateLimitError, DaytonaTimeoutError

    from autogen.coding import DaytonaCodeExecutor, DaytonaCodeResult
    from autogen.coding.daytona_code_executor import DaytonaSandboxResources

    _has_daytona = DaytonaCodeExecutor is not None
except ImportError:
    _has_daytona = False

pytestmark = pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sandbox():
    """A fully pre-configured mock Daytona sandbox."""
    sandbox = Mock()
    sandbox.id = "test-sandbox-id"
    sandbox.process.code_run.return_value = Mock(exit_code=0, result="")
    sandbox.process.exec.return_value = Mock(exit_code=0, result="")
    sandbox.fs.upload_file = Mock()
    sandbox.fs.delete_file = Mock()
    sandbox.delete = Mock()
    return sandbox


@pytest.fixture
def executor(mock_sandbox):
    """A DaytonaCodeExecutor with all external calls mocked out."""
    with (
        patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
        patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
        patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
        patch("atexit.register"),
    ):
        mock_daytona_cls.return_value.create.return_value = mock_sandbox
        yield DaytonaCodeExecutor(api_key="test-key")


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaCodeExecutorInit:
    def _make_executor(self, mock_sandbox, **kwargs):
        """Helper: create an executor with all SDK calls mocked."""
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            return DaytonaCodeExecutor(**kwargs)

    def test_default_init(self, mock_sandbox):
        """Executor creates a sandbox with no explicit config."""
        executor = self._make_executor(mock_sandbox)
        assert executor._sandbox is mock_sandbox
        assert executor._timeout == 60
        assert executor._snapshot is None
        assert executor._image is None
        assert executor._env_vars == {}
        assert executor._resources is None

    def test_init_with_api_key(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona"),
            patch("autogen.coding.daytona_code_executor.DaytonaConfig") as mock_cfg,
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_cfg.return_value = Mock()
            DaytonaCodeExecutor(api_key="my-key")
            mock_cfg.assert_called_once_with(api_key="my-key")

    def test_init_with_all_connection_params(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona"),
            patch("autogen.coding.daytona_code_executor.DaytonaConfig") as mock_cfg,
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_cfg.return_value = Mock()
            DaytonaCodeExecutor(api_key="k", api_url="https://example.com", target="eu")
            mock_cfg.assert_called_once_with(api_key="k", api_url="https://example.com", target="eu")

    def test_init_omits_none_connection_params(self, mock_sandbox):
        """None params are not forwarded to DaytonaConfig — SDK reads env vars for them."""
        with (
            patch("autogen.coding.daytona_code_executor.Daytona"),
            patch("autogen.coding.daytona_code_executor.DaytonaConfig") as mock_cfg,
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_cfg.return_value = Mock()
            DaytonaCodeExecutor()
            mock_cfg.assert_called_once_with()  # no kwargs

    def test_init_with_timeout(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox, timeout=120)
        assert executor._timeout == 120

    def test_init_with_snapshot(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox, snapshot="my-snapshot")
        assert executor._snapshot == "my-snapshot"

    def test_init_with_image_string(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox, image="python:3.12")
        assert executor._image == "python:3.12"

    def test_init_with_name(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox, name="my-agent")
        assert executor._name == "my-agent"

    def test_init_auto_generates_name_when_none(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox)
        assert executor._name.startswith("ag2-")
        assert len(executor._name) == len("ag2-") + 8  # ag2- + 8 hex chars

    def test_init_with_env_vars(self, mock_sandbox):
        executor = self._make_executor(mock_sandbox, env_vars={"FOO": "bar", "KEY": "secret"})
        assert executor._env_vars == {"FOO": "bar", "KEY": "secret"}

    def test_init_with_resources(self, mock_sandbox):
        resources = DaytonaSandboxResources(cpu=2, memory=4, disk=4)
        executor = self._make_executor(mock_sandbox, resources=resources)
        assert executor._resources is resources

    def test_invalid_timeout_raises(self):
        with (
            pytest.raises(ValueError, match="Timeout must be greater than or equal to 1"),
            patch("autogen.coding.daytona_code_executor.Daytona"),
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
        ):
            DaytonaCodeExecutor(timeout=0)

    def test_snapshot_and_image_both_raises(self):
        with (
            pytest.raises(ValueError, match="Cannot specify both"),
            patch("autogen.coding.daytona_code_executor.Daytona"),
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
        ):
            DaytonaCodeExecutor(snapshot="snap", image="python:3.12")

    def test_sandbox_creation_failure_raises_runtime_error(self):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.side_effect = Exception("network error")
            with pytest.raises(RuntimeError, match="Failed to create Daytona sandbox"):
                DaytonaCodeExecutor(api_key="test-key")

    def test_sandbox_creation_timeout_raises_runtime_error(self):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.side_effect = DaytonaTimeoutError("sandbox start timed out")
            with pytest.raises(RuntimeError, match="timed out"):
                DaytonaCodeExecutor(api_key="test-key")

    def test_sandbox_creation_rate_limit_raises_runtime_error(self):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.side_effect = DaytonaRateLimitError("quota exceeded")
            with pytest.raises(RuntimeError, match="rate limit"):
                DaytonaCodeExecutor(api_key="test-key")

    def test_sandbox_creation_daytona_error_raises_runtime_error(self):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.side_effect = DaytonaError("api error")
            with pytest.raises(RuntimeError, match="Failed to create Daytona sandbox"):
                DaytonaCodeExecutor(api_key="test-key")

    def test_atexit_registered(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register") as mock_atexit,
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            executor = DaytonaCodeExecutor(api_key="test-key")
            mock_atexit.assert_called_once_with(executor.delete)


# ---------------------------------------------------------------------------
# _create_sandbox — sandbox params routing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestCreateSandbox:
    def test_default_uses_snapshot_params_without_snapshot_arg(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams") as mock_snap_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            executor = DaytonaCodeExecutor(api_key="k", env_vars={"A": "1"}, name="test")
            mock_snap_params.assert_called_once_with(
                env_vars={"A": "1"},
                name=executor._name,
                auto_stop_interval=0,
            )

    def test_snapshot_uses_snapshot_params_with_snapshot_arg(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams") as mock_snap_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            executor = DaytonaCodeExecutor(api_key="k", snapshot="my-snap", name="test")
            mock_snap_params.assert_called_once_with(
                snapshot="my-snap",
                env_vars={},
                name=executor._name,
                auto_stop_interval=0,
            )

    def test_image_string_uses_image_params(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams") as mock_img_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            executor = DaytonaCodeExecutor(api_key="k", image="python:3.12-slim", name="test")
            mock_img_params.assert_called_once_with(
                image="python:3.12-slim",
                env_vars={},
                name=executor._name,
                resources=None,
                auto_stop_interval=0,
            )

    def test_image_object_uses_image_params(self, mock_sandbox):
        from daytona import Image

        declarative_image = Image.base("python:3.12-slim")
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams") as mock_img_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            DaytonaCodeExecutor(api_key="k", image=declarative_image)
            call_kwargs = mock_img_params.call_args.kwargs
            assert call_kwargs["image"] is declarative_image
            assert call_kwargs["auto_stop_interval"] == 0

    def test_image_with_resources_passes_sdk_resources(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams") as mock_img_params,
            patch("autogen.coding.daytona_code_executor.Resources") as mock_resources_cls,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            mock_sdk_resources = Mock()
            mock_resources_cls.return_value = mock_sdk_resources
            resources = DaytonaSandboxResources(cpu=2, memory=4, disk=4)
            DaytonaCodeExecutor(api_key="k", image="python:3.12", resources=resources)
            mock_resources_cls.assert_called_once_with(cpu=2, memory=4, disk=4)
            call_kwargs = mock_img_params.call_args.kwargs
            assert call_kwargs["resources"] is mock_sdk_resources

    def test_resources_not_forwarded_without_any_field_set(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams") as mock_img_params,
            patch("autogen.coding.daytona_code_executor.Resources") as mock_resources_cls,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            # DaytonaSandboxResources with all None fields — should not build SDK Resources
            DaytonaCodeExecutor(api_key="k", image="python:3.12", resources=DaytonaSandboxResources())
            mock_resources_cls.assert_not_called()
            assert mock_img_params.call_args.kwargs["resources"] is None

    def test_auto_stop_interval_always_zero_snapshot_path(self, mock_sandbox):
        """Sandbox must never auto-stop — always pass auto_stop_interval=0."""
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams") as mock_snap_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            DaytonaCodeExecutor(api_key="k")
            assert mock_snap_params.call_args.kwargs["auto_stop_interval"] == 0

    def test_auto_stop_interval_always_zero_image_path(self, mock_sandbox):
        """auto_stop_interval=0 must also be passed on the image code path."""
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromImageParams") as mock_img_params,
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            DaytonaCodeExecutor(api_key="k", image="python:3.12")
            assert mock_img_params.call_args.kwargs["auto_stop_interval"] == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaCodeExecutorProperties:
    def test_code_extractor_is_markdown(self, executor):
        assert isinstance(executor.code_extractor, MarkdownCodeExtractor)

    def test_timeout_property(self, executor):
        assert executor.timeout == 60


# ---------------------------------------------------------------------------
# Language normalisation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestNormalizeLanguage:
    def test_python_variants(self, executor):
        assert executor._normalize_language("python") == "python"
        assert executor._normalize_language("py") == "python"
        assert executor._normalize_language("Python") == "python"
        assert executor._normalize_language("PYTHON") == "python"

    def test_javascript_variants(self, executor):
        assert executor._normalize_language("javascript") == "javascript"
        assert executor._normalize_language("js") == "javascript"
        assert executor._normalize_language("JavaScript") == "javascript"

    def test_typescript_variants(self, executor):
        assert executor._normalize_language("typescript") == "typescript"
        assert executor._normalize_language("ts") == "typescript"

    def test_bash_variants(self, executor):
        assert executor._normalize_language("bash") == "bash"
        assert executor._normalize_language("shell") == "bash"
        assert executor._normalize_language("sh") == "sh"  # sh is its own canonical name

    def test_unknown_passthrough(self, executor):
        assert executor._normalize_language("java") == "java"
        assert executor._normalize_language("rust") == "rust"


# ---------------------------------------------------------------------------
# execute_code_blocks
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestExecuteCodeBlocks:
    def test_empty_blocks_returns_success(self, executor, mock_sandbox):
        result = executor.execute_code_blocks([])
        assert result.exit_code == 0
        assert result.output == ""
        assert result.sandbox_id == mock_sandbox.id

    def test_unsupported_language_returns_error(self, executor, mock_sandbox):
        result = executor.execute_code_blocks([CodeBlock(code="...", language="java")])
        assert result.exit_code == 1
        assert "Unsupported language: 'java'" in result.output
        assert result.sandbox_id == mock_sandbox.id

    def test_python_uses_code_run(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.return_value = Mock(exit_code=0, result="hello")
        result = executor.execute_code_blocks([CodeBlock(code="print('hello')", language="python")])
        mock_sandbox.process.code_run.assert_called_once_with("print('hello')", timeout=60)
        assert result.exit_code == 0
        assert result.output == "hello"

    def test_bash_uploads_file_and_execs(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="done")
        result = executor.execute_code_blocks([CodeBlock(code="echo done", language="bash")])
        mock_sandbox.fs.upload_file.assert_called_once()
        upload_args = mock_sandbox.fs.upload_file.call_args
        assert upload_args[0][0] == b"echo done"
        assert upload_args[0][1].endswith(".sh")
        assert "bash" in mock_sandbox.process.exec.call_args[0][0]
        assert result.exit_code == 0

    def test_javascript_uses_node(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="42")
        executor.execute_code_blocks([CodeBlock(code="console.log(42)", language="javascript")])
        cmd = mock_sandbox.process.exec.call_args[0][0]
        assert cmd.startswith("node ")
        assert cmd.endswith(".js")

    def test_typescript_uses_ts_node(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="")
        executor.execute_code_blocks([CodeBlock(code="console.log('hi')", language="typescript")])
        cmd = mock_sandbox.process.exec.call_args[0][0]
        assert "ts-node" in cmd
        assert "--transpile-only" in cmd
        assert cmd.endswith(".ts")

    def test_script_file_cleaned_up_after_exec(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="")
        executor.execute_code_blocks([CodeBlock(code="echo hi", language="bash")])
        mock_sandbox.fs.delete_file.assert_called_once()

    def test_script_cleanup_failure_does_not_propagate(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="ok")
        mock_sandbox.fs.delete_file.side_effect = Exception("fs error")
        # Should not raise — cleanup failure is swallowed
        result = executor.execute_code_blocks([CodeBlock(code="echo hi", language="bash")])
        assert result.exit_code == 0

    def test_nonzero_exit_code_returns_early(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.return_value = Mock(exit_code=1, result="NameError: x")
        result = executor.execute_code_blocks([CodeBlock(code="print(x)", language="python")])
        assert result.exit_code == 1
        assert result.output == "NameError: x"

    def test_daytona_error_during_exec_returns_error(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = DaytonaError("sandbox unreachable")
        result = executor.execute_code_blocks([CodeBlock(code="print('hi')", language="python")])
        assert result.exit_code == 1
        assert "sandbox unreachable" in result.output

    def test_timeout_error_during_exec_returns_error(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = DaytonaTimeoutError("execution timed out")
        result = executor.execute_code_blocks([CodeBlock(code="import time; time.sleep(999)", language="python")])
        assert result.exit_code == 1
        assert "timed out" in result.output

    def test_rate_limit_error_during_exec_returns_error(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = DaytonaRateLimitError("rate limit exceeded")
        result = executor.execute_code_blocks([CodeBlock(code="print('hi')", language="python")])
        assert result.exit_code == 1
        assert "rate limit" in result.output.lower()

    def test_unexpected_error_during_exec_returns_error(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = RuntimeError("unexpected failure")
        result = executor.execute_code_blocks([CodeBlock(code="print('hi')", language="python")])
        assert result.exit_code == 1
        assert "unexpected failure" in result.output

    def test_multiple_blocks_all_success_joins_output(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = [
            Mock(exit_code=0, result="first"),
            Mock(exit_code=0, result="second"),
        ]
        result = executor.execute_code_blocks([
            CodeBlock(code="print('first')", language="python"),
            CodeBlock(code="print('second')", language="python"),
        ])
        assert result.exit_code == 0
        assert result.output == "first\nsecond"

    def test_multiple_blocks_stops_on_first_failure(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.side_effect = [
            Mock(exit_code=1, result="error in block 1"),
            Mock(exit_code=0, result="should not run"),
        ]
        result = executor.execute_code_blocks([
            CodeBlock(code="bad code", language="python"),
            CodeBlock(code="good code", language="python"),
        ])
        assert result.exit_code == 1
        assert result.output == "error in block 1"
        assert mock_sandbox.process.code_run.call_count == 1

    def test_sandbox_id_included_in_result(self, executor, mock_sandbox):
        mock_sandbox.process.code_run.return_value = Mock(exit_code=0, result="ok")
        result = executor.execute_code_blocks([CodeBlock(code="x=1", language="python")])
        assert result.sandbox_id == "test-sandbox-id"

    def test_language_alias_resolved(self, executor, mock_sandbox):
        """'py' fence alias should be treated as python and use code_run."""
        mock_sandbox.process.code_run.return_value = Mock(exit_code=0, result="")
        executor.execute_code_blocks([CodeBlock(code="x=1", language="py")])
        mock_sandbox.process.code_run.assert_called_once()

    def test_sh_command_used_for_sh_language(self, executor, mock_sandbox):
        mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="")
        executor.execute_code_blocks([CodeBlock(code="echo hi", language="sh")])
        cmd = mock_sandbox.process.exec.call_args[0][0]
        assert cmd.startswith("sh ")
        assert not cmd.startswith("bash ")
        assert cmd.endswith(".sh")

    def test_timeout_forwarded_to_code_run(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            mock_sandbox.process.code_run.return_value = Mock(exit_code=0, result="")
            DaytonaCodeExecutor(api_key="k", timeout=30).execute_code_blocks([CodeBlock(code="x=1", language="python")])
            mock_sandbox.process.code_run.assert_called_once_with("x=1", timeout=30)

    def test_timeout_forwarded_to_exec(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            mock_sandbox.process.exec.return_value = Mock(exit_code=0, result="")
            DaytonaCodeExecutor(api_key="k", timeout=45).execute_code_blocks([
                CodeBlock(code="echo hi", language="bash")
            ])
            _, kwargs = mock_sandbox.process.exec.call_args
            assert kwargs.get("timeout") == 45


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestRestart:
    def test_restart_deletes_old_sandbox_and_creates_new(self, executor, mock_sandbox):
        new_sandbox = Mock()
        new_sandbox.id = "new-sandbox-id"
        with patch.object(executor, "_create_sandbox", return_value=new_sandbox) as mock_create:
            executor.restart()
            mock_sandbox.delete.assert_called_once()
            mock_create.assert_called_once()
            assert executor._sandbox is new_sandbox

    def test_restart_continues_if_delete_fails(self, executor, mock_sandbox):
        mock_sandbox.delete.side_effect = Exception("already gone")
        new_sandbox = Mock()
        with patch.object(executor, "_create_sandbox", return_value=new_sandbox):
            executor.restart()  # should not raise
            assert executor._sandbox is new_sandbox


# ---------------------------------------------------------------------------
# delete / lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestLifecycle:
    def test_delete_calls_sandbox_delete(self, executor, mock_sandbox):
        executor.delete()
        mock_sandbox.delete.assert_called_once()

    def test_delete_sets_sandbox_to_none(self, executor, mock_sandbox):
        executor.delete()
        assert executor._sandbox is None

    def test_delete_is_idempotent(self, executor, mock_sandbox):
        executor.delete()
        executor.delete()  # second call should not raise
        mock_sandbox.delete.assert_called_once()  # underlying only called once

    def test_delete_swallows_sandbox_exception(self, executor, mock_sandbox):
        mock_sandbox.delete.side_effect = Exception("already deleted")
        executor.delete()  # should not raise

    def test_delete_unregisters_atexit(self, executor):
        with patch("atexit.unregister") as mock_unregister:
            executor.delete()
            mock_unregister.assert_called_once_with(executor.delete)

    def test_delete_unregisters_atexit_on_context_manager_exit(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
            patch("atexit.unregister") as mock_unregister,
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            with DaytonaCodeExecutor(api_key="k") as executor:
                pass
            mock_unregister.assert_called_with(executor.delete)

    def test_context_manager_enter_returns_executor(self, executor):
        result = executor.__enter__()
        assert result is executor

    def test_context_manager_exit_calls_delete(self, executor):
        with patch.object(executor, "delete") as mock_delete:
            executor.__exit__(None, None, None)
            mock_delete.assert_called_once()

    def test_context_manager_exit_called_on_exception(self, mock_sandbox):
        with (
            patch("autogen.coding.daytona_code_executor.Daytona") as mock_daytona_cls,
            patch("autogen.coding.daytona_code_executor.DaytonaConfig"),
            patch("autogen.coding.daytona_code_executor.CreateSandboxFromSnapshotParams"),
            patch("atexit.register"),
            pytest.raises(ValueError),
        ):
            mock_daytona_cls.return_value.create.return_value = mock_sandbox
            with DaytonaCodeExecutor(api_key="k"):
                raise ValueError("test error")
        mock_sandbox.delete.assert_called_once()


# ---------------------------------------------------------------------------
# DaytonaCodeResult
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaCodeResult:
    def test_creation_with_sandbox_id(self):
        result = DaytonaCodeResult(exit_code=0, output="hello", sandbox_id="abc-123")
        assert result.exit_code == 0
        assert result.output == "hello"
        assert result.sandbox_id == "abc-123"

    def test_sandbox_id_defaults_to_none(self):
        result = DaytonaCodeResult(exit_code=1, output="error")
        assert result.sandbox_id is None

    def test_inherits_from_code_result(self):
        from autogen.coding.base import CodeResult

        result = DaytonaCodeResult(exit_code=0, output="")
        assert isinstance(result, CodeResult)


# ---------------------------------------------------------------------------
# DaytonaSandboxResources
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestSupportedLanguages:
    def test_supported_languages_constant(self):
        assert set(DaytonaCodeExecutor.SUPPORTED_LANGUAGES) == {"python", "bash", "sh", "javascript", "typescript"}

    def test_supported_languages_are_all_lowercase(self):
        for lang in DaytonaCodeExecutor.SUPPORTED_LANGUAGES:
            assert lang == lang.lower()


@pytest.mark.skipif(not _has_daytona, reason="Daytona dependencies not installed")
class TestDaytonaSandboxResources:
    def test_all_fields_default_to_none(self):
        r = DaytonaSandboxResources()
        assert r.cpu is None
        assert r.memory is None
        assert r.disk is None

    def test_fields_can_be_set(self):
        r = DaytonaSandboxResources(cpu=2, memory=4, disk=4)
        assert r.cpu == 2
        assert r.memory == 4
        assert r.disk == 4

    def test_partial_fields(self):
        r = DaytonaSandboxResources(memory=4)
        assert r.cpu is None
        assert r.memory == 4
        assert r.disk is None
