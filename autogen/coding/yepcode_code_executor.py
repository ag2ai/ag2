# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""YepCode code executor implementation."""

import os
from collections.abc import Callable
from typing import ClassVar

from pydantic import Field

from ..doc_utils import export_module
from .base import CodeBlock, CodeExecutor, CodeExtractor, CodeResult
from .markdown_code_extractor import MarkdownCodeExtractor

try:
    from dotenv import load_dotenv

    _load_dotenv: Callable[[], bool] | None = load_dotenv
except ImportError:
    _load_dotenv = None

try:
    from yepcode_run import YepCodeApiConfig, YepCodeRun
except ImportError:
    YepCodeRun = None
    YepCodeApiConfig = None


@export_module("autogen.coding")
class YepCodeCodeResult(CodeResult):
    """A code result class for YepCode executor."""

    execution_id: str | None = Field(default=None, description="The YepCode execution ID for this result.")


@export_module("autogen.coding")
class YepCodeCodeExecutor(CodeExecutor):
    """A code executor class that executes code using YepCode's serverless runtime.

    This executor runs code in YepCode's secure, production-grade sandboxes.
    It supports Python and JavaScript execution with access to any external library with automatic discovery and installation.

    The executor executes code blocks serially in the order they are received.
    Each code block is executed in a separate YepCode execution environment.
    Currently supports Python and JavaScript languages.

    Args:
        api_token (Optional[str]): YepCode API token. If None, will try to get from YEPCODE_API_TOKEN environment variable.
        timeout (int): The timeout for code execution in seconds. Default is 60.
        remove_on_done (bool): Whether to remove the execution after completion. Default is False.
        sync_execution (bool): Whether to wait for execution to complete. Default is True.

    Raises:
        ImportError: If yepcode-run package is not installed.
        ValueError: If YepCode API token is not provided or timeout is invalid.
        RuntimeError: If YepCode runner initialization fails.
    """

    SUPPORTED_LANGUAGES: ClassVar[list[str]] = ["python", "javascript"]

    def __init__(
        self,
        api_token: str | None = None,
        timeout: int = 60,
        remove_on_done: bool = False,
        sync_execution: bool = True,
    ):
        if YepCodeRun is None or YepCodeApiConfig is None:
            raise ImportError(
                "Missing dependencies for YepCodeCodeExecutor. Please install with: pip install ag2[yepcode]"
            )

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        # Load environment variables from .env file if dotenv is available
        if _load_dotenv is not None:
            _load_dotenv()

        # Get API token from parameter or environment
        self._api_token = api_token or os.getenv("YEPCODE_API_TOKEN")
        if not self._api_token:
            raise ValueError(
                "YepCode API token is required. Provide it via api_token parameter or YEPCODE_API_TOKEN environment variable."
            )

        self._timeout = timeout
        self._remove_on_done = remove_on_done
        self._sync_execution = sync_execution

        try:
            config = YepCodeApiConfig(api_token=self._api_token)
            self._runner = YepCodeRun(config)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YepCode runner: {str(e)}") from e

    @property
    def code_extractor(self) -> CodeExtractor:
        """(Experimental) Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    @property
    def timeout(self) -> int:
        """The timeout for code execution."""
        return self._timeout

    def _normalize_language(self, language: str) -> str:
        """Normalize language name to YepCode format."""
        lang = language.lower()
        if lang in ["js", "javascript"]:
            return "javascript"
        elif lang in ["python", "py"]:
            return "python"
        else:
            return lang

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> YepCodeCodeResult:
        """Execute the code blocks and return the result.

        Args:
            code_blocks (List[CodeBlock]): The code blocks to execute.

        Returns:
            YepCodeCodeResult: The result of the code execution.
        """
        if not code_blocks:
            return YepCodeCodeResult(exit_code=0, output="")

        outputs: list[str] = []
        last_execution_id: str | None = None

        for code_block in code_blocks:
            lang = self._normalize_language(code_block.language)

            if lang not in ["python", "javascript"]:
                return YepCodeCodeResult(
                    exit_code=1,
                    output=f"Unsupported language: {code_block.language}. Supported languages: {', '.join(self.SUPPORTED_LANGUAGES)}",
                )

            try:
                # Execute code using YepCode
                execution = self._runner.run(
                    code_block.code,
                    {
                        "language": lang,
                        "removeOnDone": self._remove_on_done,
                        "timeout": self._timeout * 1000,  # Convert to milliseconds
                    },
                )

                last_execution_id = execution.id

                if self._sync_execution:
                    # Wait for execution to complete
                    execution.wait_for_done()

                    logs_output = ""
                    # Get logs
                    if execution.logs:
                        logs_output = "\n\nExecution logs:\n" + "\n".join([
                            f"{log.timestamp} - {log.level}: {log.message}" for log in execution.logs
                        ])

                    # Check if execution was successful
                    if execution.error:
                        output = f"Execution failed with error:\n{execution.error}{logs_output}"

                        return YepCodeCodeResult(exit_code=1, output=output, execution_id=execution.id)

                    # Get output
                    output = ""
                    if execution.return_value:
                        output = f"Execution result:\n{execution.return_value}"

                    output += logs_output

                    outputs.append(output)
                else:
                    outputs.append(f"Execution started with ID: {execution.id}")

            except Exception as e:
                return YepCodeCodeResult(
                    exit_code=1,
                    output=f"Error executing code: {str(e)}",
                    execution_id=last_execution_id,
                )

        return YepCodeCodeResult(exit_code=0, output="\n===\n".join(outputs), execution_id=last_execution_id)

    def restart(self) -> None:
        """Restart the code executor."""
        pass
