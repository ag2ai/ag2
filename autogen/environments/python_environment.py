# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import Any, Optional

__all__ = ["PythonEnvironment"]


class PythonEnvironment(ABC):
    """Python execution environments base class"""

    # Shared context variable for tracking the current environment
    _current_python_environment: ContextVar["PythonEnvironment"] = ContextVar("_current_python_environment")

    def __init__(self):
        """
        Initialize the Python environment.
        """
        self._token = None

    def __enter__(self):
        """
        Enter the environment context.
        Sets this environment as the current one.
        """
        # Set up the environment
        self._setup_environment()

        # Set this as the current Python environment in the context
        self._token = PythonEnvironment._current_python_environment.set(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the environment context.
        Resets the current environment and performs cleanup.
        """
        # Reset the context variable if this was the active environment
        if self._token is not None:
            PythonEnvironment._current_python_environment.reset(self._token)
            self._token = None

        # Clean up resources
        self._cleanup_environment()

    @abstractmethod
    def _setup_environment(self) -> None:
        """Set up the Python environment. Called by __enter__."""
        pass

    @abstractmethod
    def _cleanup_environment(self) -> None:
        """Clean up the Python environment. Called by __exit__."""
        pass

    @abstractmethod
    def get_executable(self) -> str:
        """
        Get the path to the Python executable in this environment.

        Returns:
            The full path to the Python executable.
        """
        pass

    @abstractmethod
    async def execute_code(self, code: str, script_path: str, timeout: int = 30) -> dict[str, Any]:
        """
        Execute the given code in this environment.

        Args:
            code: The Python code to execute.
            script_path: Path where the code should be saved before execution.
            timeout: Maximum execution time in seconds.

        Returns:
            dict with execution results including stdout, stderr, and success status.
        """
        pass

    # Utility method for subclasses to wrap (for async support)
    def _write_to_file(self, script_path: str, content: str) -> None:
        """
        Write content to a file (blocking operation).

        This is a helper method for use with asyncify in async contexts.

        Args:
            script_path: Path to the file to write.
            content: Content to write to the file.
        """
        with open(script_path, "w") as f:
            f.write(content)

    # Utility method for subclasses to wrap (for async support)
    def _run_subprocess(self, cmd: list[str], timeout: int) -> subprocess.CompletedProcess:
        """
        Run a subprocess (blocking operation).

        This is a helper method for use with asyncify in async contexts.

        Args:
            cmd: Command to run as a list of strings.
            timeout: Maximum execution time in seconds.

        Returns:
            CompletedProcess instance with results of the subprocess.
        """
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    @classmethod
    def get_current_python_environment(
        cls, python_environment: Optional["PythonEnvironment"] = None
    ) -> Optional["PythonEnvironment"]:
        """
        Get the current Python environment or the specified one if provided.

        Args:
            python_environment: Optional environment to return if specified.

        Returns:
            The current Python environment or None if none is active.
        """
        if python_environment is not None:
            return python_environment
        try:
            return cls._current_python_environment.get()
        except LookupError:
            return None
