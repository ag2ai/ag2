import sys

import pytest

# This marker will skip all tests in this file if the Python version is less than 3.11.
# This is crucial because the yepcode-run dependency itself requires Python 3.11+.
pytestmark = pytest.mark.skipif(sys.version_info < (3, 11), reason="YepCode requires Python 3.11 or higher")

# We only import the executor if the Python version is sufficient.
# This prevents an ImportError during pytest's test collection phase on Python 3.10.
if sys.version_info >= (3, 11):
    from autogen.coding.base import CodeBlock
    from autogen.coding.yepcode_code_executor import YepCodeCodeExecutor


class TestYepCodeCodeExecutor:
    """Tests for YepCodeCodeExecutor."""

    def test_init_with_api_token(self):
        """Test initialization with an API token."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        assert executor.api_token == "test_token"

    def test_init_with_environment_token(self, monkeypatch):
        """Test initialization with an environment variable for the API token."""
        monkeypatch.setenv("YEPCODE_API_TOKEN", "env_token")
        executor = YepCodeCodeExecutor()
        assert executor.api_token == "env_token"

    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        executor = YepCodeCodeExecutor(api_token="test_token", timeout=120, remove_on_done=True, sync_execution=False)
        assert executor.timeout == 120
        assert executor.remove_on_done is True
        assert executor.sync_execution is False

    def test_init_with_invalid_timeout(self):
        """Test initialization with invalid timeout raises ValueError."""
        with pytest.raises(ValueError, match="Timeout must be greater than or equal to 1"):
            YepCodeCodeExecutor(api_token="test_token", timeout=0)

    def test_init_runner_failure(self):
        """Test initialization failure when no API token is provided."""
        with pytest.raises(ValueError, match="YepCode API token is required"):
            YepCodeCodeExecutor()

    def test_code_extractor_property(self):
        """Test the code_extractor property."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        assert executor.code_extractor is not None

    def test_timeout_property(self):
        """Test the timeout property."""
        executor = YepCodeCodeExecutor(api_token="test_token", timeout=90)
        assert executor.timeout == 90

    def test_normalize_language(self):
        """Test language normalization."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        assert executor.normalize_language("python") == "python"
        assert executor.normalize_language("py") == "python"
        assert executor.normalize_language("javascript") == "javascript"
        assert executor.normalize_language("js") == "javascript"
        assert executor.normalize_language("bash") == "shell"
        assert executor.normalize_language("sh") == "shell"
        assert executor.normalize_language("shell") == "shell"
        assert executor.normalize_language("unknown") == "unknown"

    def test_execute_empty_code_blocks(self):
        """Test execution with no code blocks."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        result = executor.execute_code_blocks([])
        assert result.exit_code == 0
        assert result.output == ""

    def test_execute_unsupported_language(self):
        """Test execution with an unsupported language."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(code="console.log('hello');", language="typescript")]
        result = executor.execute_code_blocks(code_blocks)
        assert result.exit_code == 1
        assert "Language 'typescript' is not supported" in result.output

    def test_execute_successful_python_code(self, monkeypatch):
        """Test successful execution of Python code."""

        # Mock the YepCodeRun class
        class MockYepCodeRun:
            def __init__(self, api_config):
                pass

            def run(self, code, lang):
                return {"output": "hello world", "error": ""}

        monkeypatch.setattr("autogen.coding.yepcode_code_executor.YepCodeRun", MockYepCodeRun)
        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(code="print('hello world')", language="python")]
        result = executor.execute_code_blocks(code_blocks)
        assert result.exit_code == 0
        assert result.output == "hello world"

    def test_execute_code_with_error(self, monkeypatch):
        """Test execution of code that results in an error."""

        class MockYepCodeRun:
            def __init__(self, api_config):
                pass

            def run(self, code, lang):
                return {"output": "", "error": "SyntaxError: invalid syntax"}

        monkeypatch.setattr("autogen.coding.yepcode_code_executor.YepCodeRun", MockYepCodeRun)
        executor = YepCodeCodeExecutor(api_token="test_token")
        code_blocks = [CodeBlock(code="print('hello)", language="python")]
        result = executor.execute_code_blocks(code_blocks)
        assert result.exit_code == 1
        assert result.output == "SyntaxError: invalid syntax"

    def test_restart_method(self):
        """Test that the restart method does nothing."""
        executor = YepCodeCodeExecutor(api_token="test_token")
        # Should not raise any exception
        executor.restart()


class TestYepCodeCodeResult:
    """Tests for the CodeResult class from YepCode."""

    def test_code_result_creation(self):
        """Test the creation of a CodeResult object."""
        result = YepCodeCodeExecutor.CodeResult(exit_code=0, output="output", execution_id="123")
        assert result.exit_code == 0
        assert result.output == "output"
        assert result.execution_id == "123"

    def test_code_result_without_execution_id(self):
        """Test CodeResult without an execution ID."""
        result = YepCodeCodeExecutor.CodeResult(exit_code=1, output="error")
        assert result.exit_code == 1
        assert result.output == "error"
        assert result.execution_id is None
