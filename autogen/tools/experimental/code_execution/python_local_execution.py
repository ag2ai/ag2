# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

from ....doc_utils import export_module
from ....environments import WorkingDirectory
from ....environments.python_environment import PythonEnvironment
from ... import Tool

__all__ = ["PythonLocalExecutionTool"]


@export_module("autogen.tools.experimental")
class PythonLocalExecutionTool(Tool):
    """Executes Python code locally and returns the result.

    **CAUTION**: This tool will execute code in your local environment, which can be dangerous if the code is untrusted.
    """

    def __init__(
        self,
        *,
        timeout: int = 30,
        working_directory: Optional[WorkingDirectory] = None,
        python_environment: Optional[PythonEnvironment] = None,
    ) -> None:
        """
        Initialize the PythonLocalExecutionTool.

        **CAUTION**: This tool will execute code in your local environment, which can be dangerous if the code is untrusted.

        Args:
            timeout: Maximum execution time allowed in seconds, will raise a TimeoutError exception if exceeded.
            working_directory: Optional WorkingDirectory context manager to use.
            python_environment: Optional PythonEnvironment to use. If None, will auto-detect or create based on other parameters.
        """
        # Store configuration parameters
        self.timeout = timeout
        self.working_directory = WorkingDirectory.get_current_working_directory(working_directory)
        tool_python_environment = PythonEnvironment.get_current_python_environment(python_environment)

        assert self.working_directory, "No Working directory found"
        assert tool_python_environment, "No Python environment found"

        self.python_environment = tool_python_environment

        # Pydantic model to contain the code and list of libraries to execute
        class CodeExecutionRequest(BaseModel):
            code: Annotated[str, Field(description="Python code to execute")]
            libraries: Annotated[list[str], Field(description="List of libraries to install before execution")]

        # The tool function, this is what goes to the LLM
        async def execute_python_locally(
            code_execution_request: Annotated[CodeExecutionRequest, "Python code and the libraries required"],
        ) -> dict[str, Any]:
            """
            Executes Python code locally and returns the result.

            Args:
                code_execution_request (CodeExecutionRequest): The Python code and libraries to execute
            """
            code = code_execution_request.code
            # We still accept libraries in the request but don't pre-install them
            # This allows the code to reference which libraries it needs

            # Prepare a script file path
            script_dir = self._get_script_directory()
            script_path = os.path.join(script_dir, "script.py")

            # Execute the code
            return await self.python_environment.execute_code(code=code, script_path=script_path, timeout=self.timeout)

        super().__init__(
            name="python_execute_local",
            description="Executes Python code locally and returns the result.",
            func_or_tool=execute_python_locally,
        )

    def _get_script_directory(self) -> str:
        """Get the directory to use for scripts."""
        if self.working_directory and hasattr(self.working_directory, "path") and self.working_directory.path:
            path = self.working_directory.path
            os.makedirs(path, exist_ok=True)
            return path
        return tempfile.mkdtemp(prefix="ag2_script_dir_")
