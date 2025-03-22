# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from typing import Any, Optional

from asyncer import asyncify

from .python_environment import PythonEnvironment

__all__ = ["SystemPythonEnvironment"]


class SystemPythonEnvironment(PythonEnvironment):
    """A Python environment using the system's Python installation."""

    def __init__(
        self,
        executable: Optional[str] = None,
    ):
        """
        Initialize a system Python environment.

        Args:
            executable: Optional path to a specific Python executable. If None, uses the current Python.
        """
        super().__init__()
        self._executable = executable or sys.executable

    def _setup_environment(self) -> None:
        """Set up the system Python environment."""
        # Verify the Python executable exists
        if not os.path.exists(self._executable):
            raise RuntimeError(f"Python executable not found at: {self._executable}")

        print(f"Using system Python at: {self._executable}")

        # Verify pip is available
        try:
            pip_version = subprocess.run(
                [self._executable, "-m", "pip", "--version"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            print(f"Pip version: {pip_version.stdout.strip()}")
        except Exception as e:
            print(f"Warning: pip may not be available: {str(e)}")

    def _cleanup_environment(self) -> None:
        """Clean up the system Python environment."""
        # No cleanup needed for system Python
        pass

    def get_executable(self) -> str:
        """Get the path to the Python executable."""
        return self._executable

    async def execute_code(self, code: str, script_path: str, timeout: int = 30) -> dict[str, Any]:
        """Execute code using the system Python."""
        try:
            # Get the Python executable
            python_executable = self.get_executable()

            # Verify the executable exists
            if not os.path.exists(python_executable):
                return {"success": False, "error": f"Python executable not found at {python_executable}"}

            # Print the environment details
            print(f"Executing code with system Python: {python_executable}")
            print(f"Working directory: {os.getcwd()}")

            # Ensure the directory for the script exists
            script_dir = os.path.dirname(script_path)
            if script_dir:
                os.makedirs(script_dir, exist_ok=True)

            # Simple script header to check environment
            script_header = """
# Simple environment check
import sys
import os

# Print Python info
print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")

# Original code follows
"""
            # Write the code to the script file using asyncify (from base class)
            await asyncify(self._write_to_file)(script_path, script_header + "\n\n" + code)

            print(f"Wrote code to {script_path}")

            # Execute the script using asyncify (from base class)
            try:
                # Execute directly with subprocess using asyncify for better reliability
                result = await asyncify(self._run_subprocess)([python_executable, script_path], timeout)

                return {
                    "success": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except subprocess.TimeoutExpired:
                return {"success": False, "error": f"Execution timed out after {timeout} seconds"}

        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}
