# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

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
        venv_path: Optional[str] = None,
    ):
        """
        Initialize a virtual environment for Python execution.

        Args:
            python_version: The Python version to use (e.g., "3.11")
            venv_path: Optional path for the virtual environment. If None, creates a temp directory.
        """
        super().__init__()
        self.python_version = python_version
        self.venv_path = venv_path
        self.created_venv = False
        self._executable = None

    def _setup_environment(self) -> None:
        """Set up the virtual environment."""
        # Create a venv directory if not provided
        if self.venv_path is None:
            self.venv_path = tempfile.mkdtemp(prefix="ag2_python_env_")
            self.created_venv = True
        elif not os.path.exists(self.venv_path):
            os.makedirs(self.venv_path, exist_ok=True)
            self.created_venv = True

        # Determine the Python executable to use for creating the venv
        base_python = self._get_python_executable_for_version()

        print(f"Creating virtual environment at {self.venv_path} using {base_python}")

        # Create the virtual environment
        try:
            # Try creating venv with --system-site-packages first for more reliable package access
            _ = subprocess.run(
                [base_python, "-m", "venv", "--system-site-packages", self.venv_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Determine the Python executable in the virtual environment
            if os.name == "nt":  # Windows
                self._executable = os.path.join(self.venv_path, "Scripts", "python.exe")
                bin_dir = os.path.join(self.venv_path, "Scripts")
            else:  # Unix-like
                self._executable = os.path.join(self.venv_path, "bin", "python")
                bin_dir = os.path.join(self.venv_path, "bin")

            # Verify venv Python executable exists
            if not os.path.exists(self._executable):
                # Make sure the bin directory exists
                if not os.path.exists(bin_dir):
                    os.makedirs(bin_dir, exist_ok=True)

                # If Python executable doesn't exist, try creating a symlink
                if os.name != "nt":  # Unix-like only
                    python_link = os.path.join(bin_dir, "python")
                    if not os.path.exists(python_link):
                        os.symlink(base_python, python_link)
                        self._executable = python_link

            # Print Python path information
            print(f"Python executable: {self._executable}")

            # Simply ensure the site-packages directory exists
            site_packages_dir = None
            if os.name == "nt":  # Windows
                site_packages_dir = os.path.join(self.venv_path, "Lib", "site-packages")
            else:  # Unix-like
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
                site_packages_dir = os.path.join(self.venv_path, "lib", f"python{python_version}", "site-packages")

            os.makedirs(site_packages_dir, exist_ok=True)
            print(f"Site packages directory: {site_packages_dir}")

        except subprocess.CalledProcessError as e:
            print(f"Error creating virtual environment: {e.stderr}")
            raise RuntimeError(f"Failed to create virtual environment: {e.stderr}") from e
        except Exception as e:
            print(f"Error setting up virtual environment: {str(e)}")
            raise

    def _cleanup_environment(self) -> None:
        """Clean up the virtual environment."""
        # Note: We intentionally don't clean up the venv here to allow
        # tools to continue using it after the context exits.
        # The cleanup will need to be done explicitly or on process exit.
        print(f"Leaving virtual environment: {self.venv_path}")

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

            # Print the environment details
            print(f"Executing code with Python: {python_executable}")
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

    def _get_python_executable_for_version(self) -> str:
        """Get the Python executable for the specified version."""
        # If no specific version is requested, use the current Python
        if not self.python_version:
            return sys.executable

        # Try to find a specific Python version using pyenv if available
        try:
            pyenv_result = subprocess.run(
                ["pyenv", "which", f"python{self.python_version}"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return pyenv_result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Try common paths for Python installations
        if os.name == "nt":  # Windows
            potential_paths = [
                f"C:\\Python{self.python_version.replace('.', '')}\\python.exe",
                f"C:\\Program Files\\Python{self.python_version.replace('.', '')}\\python.exe",
                f"C:\\Program Files (x86)\\Python{self.python_version.replace('.', '')}\\python.exe",
            ]
        else:  # Unix-like
            potential_paths = [f"/usr/bin/python{self.python_version}", f"/usr/local/bin/python{self.python_version}"]

        # Try each potential path
        for path in potential_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                return path

        # If we couldn't find a specific version, use current Python and log a warning
        print(f"Warning: Python {self.python_version} not found, using {sys.executable} instead")
        return sys.executable
