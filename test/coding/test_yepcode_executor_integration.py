# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for YepCodeCodeExecutor with real API calls."""

import os
from pathlib import Path

import pytest

from autogen.coding import CodeBlock

try:
    import dotenv

    from autogen.coding import YepCodeCodeExecutor

    # Also check that the actual YepCode dependencies are available
    from autogen.coding.yepcode_code_executor import YepCodeApiConfig, YepCodeRun

    _has_yepcode = YepCodeRun is not None and YepCodeApiConfig is not None
except ImportError:
    _has_yepcode = False

pytestmark = pytest.mark.skipif(not _has_yepcode, reason="YepCode dependencies not installed")


@pytest.mark.skipif(not _has_yepcode, reason="YepCode dependencies not installed")
@pytest.mark.integration
class TestYepCodeCodeExecutorIntegration:
    """Integration test suite for YepCodeCodeExecutor with real API calls."""

    def setup_method(self):
        """Setup method run before each test."""
        # Load environment variables from .env file
        env_file = Path(__file__).parent.parent.parent / ".env"
        if env_file.exists():
            dotenv.load_dotenv(env_file)

        # Check for API token
        if not os.getenv("YEPCODE_API_TOKEN"):
            pytest.skip("YEPCODE_API_TOKEN environment variable not set (check .env file or environment)")

    def test_basic_python_execution(self):
        """Test basic Python code execution."""
        executor = YepCodeCodeExecutor(
            timeout=60,
            remove_on_done=True,  # Clean up after test
            sync_execution=True,
        )

        code_blocks = [
            CodeBlock(
                language="python",
                code="""
import datetime
import math

now = datetime.datetime.now()
print(f"Current time: {now}")
print(f"Square root of 144: {math.sqrt(144)}")

return {
    "message": "Hello from YepCode!",
    "timestamp": now.isoformat(),
    "sqrt_144": math.sqrt(144)
}
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        assert result.execution_id is not None, "Execution ID should not be None"
        assert "Current time:" in result.output, "Expected time output in result"
        assert "Square root of 144: 12.0" in result.output, "Expected math calculation in result"

    def test_javascript_execution(self):
        """Test JavaScript code execution."""
        executor = YepCodeCodeExecutor(
            timeout=60,
            remove_on_done=True,
            sync_execution=True,
        )

        code_blocks = [
            CodeBlock(
                language="javascript",
                code="""
// Test JavaScript execution with automatic npm package installation
const moment = require('moment');

console.log('Current time:', moment().format('YYYY-MM-DD HH:mm:ss'));
console.log('JavaScript execution successful!');

return {
    message: 'Hello from JavaScript!',
    timestamp: moment().toISOString(),
    packageUsed: 'moment'
};
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 0, f"Expected exit code 0, got {result.exit_code}. Output: {result.output}"
        assert result.execution_id is not None, "Execution ID should not be None"
        assert "Current time:" in result.output, "Expected time output in result"
        assert "JavaScript execution successful!" in result.output, "Expected success message in result"

    def test_error_handling(self):
        """Test error handling with invalid code."""
        executor = YepCodeCodeExecutor(
            timeout=60,
            remove_on_done=True,
            sync_execution=True,
        )

        code_blocks = [
            CodeBlock(
                language="python",
                code="""
# This code will raise a NameError
print(undefined_variable)
""",
            )
        ]

        result = executor.execute_code_blocks(code_blocks)

        assert result.exit_code == 1, f"Expected exit code 1 for error, got {result.exit_code}"
        assert result.execution_id is not None, "Execution ID should not be None"
        assert "NameError" in result.output, "Expected NameError in output"
