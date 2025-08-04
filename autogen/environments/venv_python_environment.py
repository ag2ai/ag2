# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Optional

from asyncer import asyncify

from .python_environment import PythonEnvironment

__all__ = ["VenvPythonEnvironment"]


class VenvPythonEnvironment(PythonEnvironment):
    """A Python environment using a virtual environment (venv)."""

    def __init__(
        self,
        python_version: Optional[str] = None,
        python_path: Optional[str] = None,
        venv_path: Optional[str] = None,
    ):
        """
        Initialize a virtual environment for Python execution.

        If you pass in a venv_path the path will be checked for a valid venv. If the venv doesn't exist it will be created using the python_version or python_path provided.

        If the python_version or python_path is provided and the venv_path is not, a temporary directory will be created for venv and it will be setup with the provided python version.

        If python_path is provided, it will take precedence over python_version.

        The python version will not be installed if it doesn't exist and a RuntimeError will be raised.

        Args:
            python_version: The Python version to use (e.g., "3.11"), otherwise defaults to the current executing Python version. Ignored if venv_path is provided and has a valid environment already.
            python_path: Optional direct path to a Python executable to use (must include the executable). Takes precedence over python_version if both are provided.
            venv_path: Optional path for the virtual environment, will create it if it doesn't exist. If None, creates a temp directory.
        """
        self.python_version = python_version
        self.python_path = python_path
        self.venv_path = venv_path
        self.created_venv = False
        self._executable = None
        super().__init__()

    def _setup_environment(self) -> None:
        """Set up the virtual environment."""
        # Create a venv directory if not provided
        if self.venv_path is None:
            self.venv_path = tempfile.mkdtemp(prefix="ag2_python_env_")
            self.created_venv = True

            # Determine the python version, getting it from the venv if it already has one
            base_python = self._get_python_executable_for_version()
            needs_creation = True
        else:
            # If venv_path is provided, check if it's already a valid venv
            if os.name == "nt":  # Windows
                venv_python = os.path.join(self.venv_path, "Scripts", "python.exe")
            else:  # Unix-like (Mac/Linux)
                venv_python = os.path.join(self.venv_path, "bin", "python")

            if os.path.exists(venv_python) and os.access(venv_python, os.X_OK):
                # Valid venv already exists, just use it
                self._executable = venv_python
                logging.info(f"Using existing virtual environment at {self.venv_path}")
                needs_creation = False
            else:
                # Path exists but not a valid venv, or doesn't exist
                if not os.path.exists(self.venv_path):
                    os.makedirs(self.venv_path, exist_ok=True)
                self.created_venv = True
                base_python = sys.executable
                needs_creation = True

        # Only create the venv if needed
        if needs_creation:
            logging.info(f"Creating virtual environment at {self.venv_path} using {base_python}")

            try:
                # Create the virtual environment
                _ = subprocess.run(
                    [base_python, "-m", "venv", "--system-site-packages", self.venv_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                # Determine the Python executable path
                if os.name == "nt":  # Windows
                    self._executable = os.path.join(self.venv_path, "Scripts", "python.exe")
                else:  # Unix-like (Mac/Linux)
                    self._executable = os.path.join(self.venv_path, "bin", "python")

                # Verify the executable exists
                if not os.path.exists(self._executable):
                    raise RuntimeError(
                        f"Virtual environment created but Python executable not found at {self._executable}"
                    )

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to create virtual environment: {e.stderr}") from e

    def _cleanup_environment(self) -> None:
        """Clean up the virtual environment."""
        # Note: We intentionally don't clean up the venv here to allow
        # tools to continue using it after the context exits.
        pass

    def get_executable(self) -> str:
        """Get the path to the Python executable in the virtual environment."""
        if not self._executable or not os.path.exists(self._executable):
            raise RuntimeError("Virtual environment Python executable not found")
        return self._executable

    async def execute_code(self, code: str, script_path: str, timeout: int = 30) -> dict[str, Any]:
        """Execute code in the virtual environment."""
        try:
            # Get the Python executable
            python_executable = self.get_executable()

            # Verify the executable exists
            if not os.path.exists(python_executable):
                return {"success": False, "error": f"Python executable not found at {python_executable}"}

            # Ensure the directory for the script exists
            script_dir = os.path.dirname(script_path)
            if script_dir:
                os.makedirs(script_dir, exist_ok=True)

            # Write the code to the script file using asyncify (from base class)
            await asyncify(self._write_to_file)(script_path, code)

            logging.info(f"Wrote code to {script_path}")

            try:
                # Execute directly with subprocess using asyncify for better reliability
                result = await asyncify(self._run_subprocess)([python_executable, script_path], timeout)

                # Main execution result
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

    def _get_python_executable_for_version(self) -> str:
        """Get the Python executable for the specified version and verify it can create a venv."""
        # If a specific path is provided, use it directly
        if self.python_path:
            if not os.path.exists(self.python_path) or not os.access(self.python_path, os.X_OK):
                raise RuntimeError(f"Python executable not found at {self.python_path}")
            return self.python_path

        # If no specific version is requested, use the current Python
        if not self.python_version:
            return sys.executable

        potential_executables = []

        # Try to find a specific Python version using pyenv if available
        try:
            pyenv_result = subprocess.run(
                ["pyenv", "which", f"python{self.python_version}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            potential_executables.append(pyenv_result.stdout.strip())
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Try common system paths based on platform
        if os.name == "nt":  # Windows
            potential_executables.extend([
                f"C:\\Python{self.python_version.replace('.', '')}\\python.exe",
                f"C:\\Program Files\\Python{self.python_version.replace('.', '')}\\python.exe",
                f"C:\\Program Files (x86)\\Python{self.python_version.replace('.', '')}\\python.exe",
            ])
        else:  # Unix-like (Mac and Linux)
            # Add more paths that might exist on macOS
            potential_executables.extend([
                f"/usr/bin/python{self.python_version}",
                f"/usr/local/bin/python{self.python_version}",
                f"/opt/homebrew/bin/python{self.python_version}",  # Homebrew on Apple Silicon
                f"/opt/python/bin/python{self.python_version}",
            ])

        # Try each potential path and verify it can create a venv
        for path in potential_executables:
            if os.path.exists(path) and os.access(path, os.X_OK):
                # Verify this Python can create a venv
                try:
                    test_result = subprocess.run(
                        [path, "-m", "venv", "--help"],
                        check=False,  # Don't raise exception
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5,  # Add timeout for safety
                    )
                    if test_result.returncode == 0:
                        # Successfully found a valid Python executable
                        return path
                except (subprocess.SubprocessError, FileNotFoundError):
                    continue

        # If we couldn't find the specified version, raise an exception
        raise RuntimeError(
            f"Python {self.python_version} not found or cannot create virtual environments. Provide a python_path to use a specific Python executable."
        )
