# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for YepCodeCodeExecutor."""

import os
from unittest.mock import Mock, patch

import pytest

from autogen.coding import CodeBlock, MarkdownCodeExtractor

try:
    from autogen.coding import YepCodeCodeExecutor, YepCodeCodeResult

    # Also check that the actual YepCode dependencies are available
    from autogen.coding.yepcode_code_executor import YepCodeApiConfig, YepCodeRun

    _has_yepcode = YepCodeRun is not None and YepCodeApiConfig is not None
except ImportError:
    _has_yepcode = False

pytestmark = pytest.mark.skipif(not _has_yepcode, reason="YepCode dependencies not installed")


@pytest.mark.skipif(not _has_yepcode, reason="YepCode dependencies not installed")
class TestYepCodeCodeExecutor:
    """Test suite for YepCodeCodeExecutor."""

    def setup_method(self):
        """Setup method run before each test."""
        # Clear environment variables
        if "YEPCODE_API_TOKEN" in os.environ:
            del os.environ["YEPCODE_API_TOKEN"]

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_init_with_api_token(self, mock_config, mock_runner):
        """Test initialization with API token provided."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")

        assert executor._api_token == "test_token"
        assert executor._timeout == 60
        assert executor._remove_on_done is False
        assert executor._sync_execution is True
        mock_config.assert_called_once_with(api_token="test_token")
        mock_runner.assert_called_once()

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_init_with_environment_token(self, mock_config, mock_runner):
        """Test initialization with API token from environment."""
        os.environ["YEPCODE_API_TOKEN"] = "env_token"
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor()

        assert executor._api_token == "env_token"
        mock_config.assert_called_once_with(api_token="env_token")

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_init_with_custom_parameters(self, mock_config, mock_runner):
        """Test initialization with custom parameters."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(
            api_token="test_token",
            timeout=120,
            remove_on_done=True,
            sync_execution=False,
        )

        assert executor._api_token == "test_token"
        assert executor._timeout == 120
        assert executor._remove_on_done is True
        assert executor._sync_execution is False

    def test_init_with_invalid_timeout(self):
        """Test initialization with invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be greater than or equal to 1"):
            YepCodeCodeExecutor(api_token="test_token", timeout=0)

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_init_runner_failure(self, mock_config, mock_runner):
        """Test initialization when YepCodeRun fails."""
        mock_config.return_value = Mock()
        mock_runner.side_effect = Exception("API initialization failed")

        with pytest.raises(RuntimeError, match="Failed to initialize YepCode runner"):
            YepCodeCodeExecutor(api_token="test_token")

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_code_extractor_property(self, mock_config, mock_runner):
        """Test code_extractor property returns MarkdownCodeExtractor."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")

        assert isinstance(executor.code_extractor, MarkdownCodeExtractor)

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_timeout_property(self, mock_config, mock_runner):
        """Test timeout property."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token", timeout=120)

        assert executor.timeout == 120

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_normalize_language(self, mock_config, mock_runner):
        """Test _normalize_language method."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")

        assert executor._normalize_language("python") == "python"
        assert executor._normalize_language("py") == "python"
        assert executor._normalize_language("Python") == "python"
        assert executor._normalize_language("javascript") == "javascript"
        assert executor._normalize_language("js") == "javascript"
        assert executor._normalize_language("JavaScript") == "javascript"
        assert executor._normalize_language("java") == "java"  # unsupported

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_execute_empty_code_blocks(self, mock_config, mock_runner):
        """Test execute_code_blocks with empty list."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")
        result = executor.execute_code_blocks([])

        assert result.exit_code == 0
        assert result.output == ""

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_execute_unsupported_language(self, mock_config, mock_runner):
        """Test execute_code_blocks with unsupported language."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(language="java", code="System.out.println('Hello');")]
        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 1
        assert "Unsupported language: java" in result.output

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_execute_successful_python_code(self, mock_config, mock_runner):
        """Test successful execution of Python code."""
        mock_config.return_value = Mock()
        mock_runner_instance = Mock()
        mock_runner.return_value = mock_runner_instance

        # Mock execution
        mock_execution = Mock()
        mock_execution.id = "exec_123"
        mock_execution.error = None
        mock_execution.return_value = "Hello, World!"
        mock_execution.logs = [
            Mock(
                timestamp="2023-01-01T00:00:00Z",
                level="INFO",
                message=[{"content": "Starting execution", "role": "user"}],
            )
        ]
        mock_runner_instance.run.return_value = mock_execution

        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(language="python", code="print('Hello, World!')")]
        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 0
        assert "Execution result:\nHello, World!" in result.output
        assert "Execution logs:" in result.output
        assert result.execution_id == "exec_123"

        mock_runner_instance.run.assert_called_once_with(
            "print('Hello, World!')",
            {
                "language": "python",
                "removeOnDone": False,
                "timeout": 60000,
            },
        )
        mock_execution.wait_for_done.assert_called_once()

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_execute_code_with_error(self, mock_config, mock_runner):
        """Test execution with error."""
        mock_config.return_value = Mock()
        mock_runner_instance = Mock()
        mock_runner.return_value = mock_runner_instance

        # Mock execution with error
        mock_execution = Mock()
        mock_execution.id = "exec_error"
        mock_execution.error = "NameError: name 'undefined_var' is not defined"
        mock_execution.logs = [
            Mock(
                timestamp="2023-01-01T00:00:00Z",
                level="ERROR",
                message=[{"content": "Execution failed", "role": "user"}],
            )
        ]
        mock_runner_instance.run.return_value = mock_execution

        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(language="python", code="print(undefined_var)")]
        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 1
        assert "Execution failed with error:" in result.output
        assert "NameError: name 'undefined_var' is not defined" in result.output
        assert result.execution_id == "exec_error"

    @patch("autogen.coding.yepcode_code_executor.YepCodeRun")
    @patch("autogen.coding.yepcode_code_executor.YepCodeApiConfig")
    def test_restart_method(self, mock_config, mock_runner):
        """Test restart method (currently a no-op)."""
        mock_config.return_value = Mock()
        mock_runner.return_value = Mock()

        executor = YepCodeCodeExecutor(api_token="test_token")

        # Should not raise any exception
        executor.restart()


@pytest.mark.skipif(not _has_yepcode, reason="YepCode dependencies not installed")
class TestYepCodeCodeResult:
    """Test suite for YepCodeCodeResult."""

    def test_code_result_creation(self):
        """Test YepCodeCodeResult creation."""
        result = YepCodeCodeResult(exit_code=0, output="Test output", execution_id="exec_123")

        assert result.exit_code == 0
        assert result.output == "Test output"
        assert result.execution_id == "exec_123"

    def test_code_result_without_execution_id(self):
        """Test YepCodeCodeResult creation without execution_id."""
        result = YepCodeCodeResult(exit_code=1, output="Error output")

        assert result.exit_code == 1
        assert result.output == "Error output"
        assert result.execution_id is None
