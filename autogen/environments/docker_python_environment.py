# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from typing import Any, Optional, Tuple

from asyncer import asyncify

from .python_environment import PythonEnvironment

__all__ = ["DockerPythonEnvironment"]


class DockerPythonEnvironment(PythonEnvironment):
    """A Python environment using Docker containers for isolated execution."""

    def __init__(
        self,
        image: str = "python:3.11-slim",
        container_name_prefix: str = "ag2_docker_env_",
        volumes: Optional[dict[str, str]] = None,
        environment: Optional[dict[str, str]] = None,
        network: Optional[str] = None,
        pip_packages: Optional[list[str]] = None,
        requirements_file: Optional[str] = None,
        dockerfile: Optional[str] = None,
        build_args: Optional[dict[str, str]] = None,
        cleanup_container: bool = True,
        keep_container_running: bool = False,
        container_startup_timeout: int = 30,
    ):
        """
        Initialize a Docker Python environment.

        Args:
            image: Docker image to use (ignored if dockerfile is provided)
            container_name_prefix: Prefix for container names
            volumes: Dictionary mapping host paths to container paths for mounting
            environment: Dictionary of environment variables to set in the container
            network: Docker network to attach the container to
            pip_packages: List of pip packages to install in the container
            requirements_file: Path to requirements.txt file to install in the container
            dockerfile: Optional path to a Dockerfile to build and use instead of pulling an image
            build_args: Optional build arguments for the Dockerfile
            cleanup_container: Whether to remove the container after use
            keep_container_running: Whether to keep the container running after execution
            container_startup_timeout: Timeout in seconds for container startup
        """
        self.image = image
        self.container_name_prefix = container_name_prefix
        self.volumes = volumes or {}
        self.environment = environment or {}
        self.network = network
        self.pip_packages = pip_packages or []
        self.requirements_file = requirements_file
        self.dockerfile = dockerfile
        self.build_args = build_args or {}
        self.cleanup_container = cleanup_container
        self.keep_container_running = keep_container_running
        self.container_startup_timeout = container_startup_timeout

        # Internal state
        self._container_id = None
        self._container_name = None
        self._custom_image_name = None
        self._temp_dir = None

        super().__init__()

    def _setup_environment(self) -> None:
        """Set up the Docker environment."""
        # Verify Docker is installed and accessible
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True, check=True)
            logging.info(f"Docker version: {result.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Docker not found or not accessible. Please ensure Docker is installed and running."
            ) from e

        # Create a temporary directory for file operations
        self._temp_dir = tempfile.mkdtemp(prefix="ag2_docker_")

        # Generate a unique container name
        self._container_name = f"{self.container_name_prefix}{uuid.uuid4().hex[:8]}"

        # Build custom image if Dockerfile is provided
        if self.dockerfile:
            self._build_custom_image()
        else:
            # Pull the specified image
            try:
                subprocess.run(
                    ["docker", "pull", self.image],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logging.info(f"Pulled Docker image: {self.image}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to pull Docker image: {e.stderr}") from e

        # Start the container
        self._start_container()

    def _build_custom_image(self) -> None:
        """Build a custom Docker image from the provided Dockerfile."""
        if not os.path.exists(self.dockerfile):
            raise RuntimeError(f"Dockerfile not found at: {self.dockerfile}")

        # Create a unique image name
        self._custom_image_name = f"ag2-custom-python-{uuid.uuid4().hex[:8]}"

        # Build command
        build_cmd = ["docker", "build", "-t", self._custom_image_name]

        # Add build args
        for arg_name, arg_value in self.build_args.items():
            build_cmd.extend(["--build-arg", f"{arg_name}={arg_value}"])

        # Add Dockerfile path
        build_cmd.extend(["-f", self.dockerfile, os.path.dirname(self.dockerfile)])

        try:
            logging.info(f"Building custom Docker image: {self._custom_image_name}")
            _ = subprocess.run(
                build_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            logging.info(f"Built custom Docker image: {self._custom_image_name}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build Docker image: {e.stderr}") from e

        # Use the custom image
        self.image = self._custom_image_name

    def _start_container(self) -> None:
        """Start the Docker container."""
        # Basic container run command
        run_cmd = ["docker", "run", "--name", self._container_name]

        # Add detached mode flag to run container in background
        run_cmd.append("-d")

        # Add network if specified
        if self.network:
            run_cmd.extend(["--network", self.network])

        # Add environment variables
        for env_name, env_value in self.environment.items():
            run_cmd.extend(["-e", f"{env_name}={env_value}"])

        # Add volume mounts including temp directory
        work_dir_mount = f"{self._temp_dir}:/workspace"
        run_cmd.extend(["-v", work_dir_mount])

        for host_path, container_path in self.volumes.items():
            run_cmd.extend(["-v", f"{host_path}:{container_path}"])

        # Set workspace as working directory
        run_cmd.extend(["-w", "/workspace"])

        # Add tty to keep container running
        run_cmd.append("-t")

        # Add image name
        run_cmd.append(self.image)

        # Initial command to keep container running
        run_cmd.extend(["tail", "-f", "/dev/null"])

        try:
            # Start the container
            logging.info(f"Starting Docker container: {self._container_name}")
            result = subprocess.run(
                run_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Get container ID
            self._container_id = result.stdout.strip()
            logging.info(f"Started Docker container: {self._container_name} ({self._container_id})")

            # Install pip packages if specified
            if self.pip_packages or self.requirements_file:
                self._install_packages()

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to start Docker container: {e.stderr}") from e

    def _install_packages(self) -> None:
        """Install Python packages in the running container."""
        # Install pip packages
        if self.pip_packages:
            packages_str = " ".join(self.pip_packages)
            try:
                logging.info(f"Installing pip packages: {packages_str}")
                _ = subprocess.run(
                    ["docker", "exec", self._container_name, "pip", "install", "--no-cache-dir"] + self.pip_packages,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                logging.info("Successfully installed pip packages")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Failed to install pip packages: {e.stderr}")

        # Install from requirements file
        if self.requirements_file:
            if os.path.exists(self.requirements_file):
                # Copy requirements file to temp directory
                req_filename = os.path.basename(self.requirements_file)
                temp_req_path = os.path.join(self._temp_dir, req_filename)
                shutil.copy(self.requirements_file, temp_req_path)

                try:
                    logging.info(f"Installing requirements from: {req_filename}")
                    _ = subprocess.run(
                        [
                            "docker",
                            "exec",
                            self._container_name,
                            "pip",
                            "install",
                            "--no-cache-dir",
                            "-r",
                            f"/workspace/{req_filename}",
                        ],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    logging.info("Successfully installed requirements")
                except subprocess.CalledProcessError as e:
                    logging.warning(f"Failed to install requirements: {e.stderr}")
            else:
                logging.warning(f"Requirements file not found: {self.requirements_file}")

    def _cleanup_environment(self) -> None:
        """Clean up the Docker environment."""
        if self._container_id:
            # Stop the container if it's running and we want to clean it up
            if not self.keep_container_running:
                try:
                    logging.info(f"Stopping Docker container: {self._container_name}")
                    subprocess.run(
                        ["docker", "stop", self._container_name],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except subprocess.CalledProcessError:
                    logging.warning(f"Failed to stop Docker container: {self._container_name}")

            # Remove the container if cleanup is enabled
            if self.cleanup_container and not self.keep_container_running:
                try:
                    logging.info(f"Removing Docker container: {self._container_name}")
                    subprocess.run(
                        ["docker", "rm", "-f", self._container_name],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except subprocess.CalledProcessError:
                    logging.warning(f"Failed to remove Docker container: {self._container_name}")

            # Remove the custom image if it was created
            if self._custom_image_name and self.cleanup_container:
                try:
                    logging.info(f"Removing custom Docker image: {self._custom_image_name}")
                    subprocess.run(
                        ["docker", "rmi", self._custom_image_name],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                except subprocess.CalledProcessError:
                    logging.warning(f"Failed to remove custom Docker image: {self._custom_image_name}")

        # Clean up the temporary directory
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logging.warning(f"Failed to remove temporary directory: {e}")

    def get_executable(self) -> str:
        """Get the path to the Python executable in the Docker container."""
        # This is a virtual path in the container
        return "python"

    async def execute_code(self, code: str, script_path: str, timeout: int = 30) -> dict[str, Any]:
        """Execute code in the Docker container."""
        # Ensure the container is running
        if not self._container_id:
            return {"success": False, "error": "Docker container not started"}

        try:
            # Calculate the relative path within the temp directory
            if os.path.isabs(script_path):
                rel_path = os.path.basename(script_path)
                host_script_path = os.path.join(self._temp_dir, rel_path)
            else:
                rel_path = script_path
                host_script_path = os.path.join(self._temp_dir, rel_path)

            # Ensure the directory for the script exists
            script_dir = os.path.dirname(host_script_path)
            if script_dir:
                os.makedirs(script_dir, exist_ok=True)

            # Write the code to the script file on the host
            await asyncify(self._write_to_file)(host_script_path, code)

            # Path to the script in the container
            container_script_path = f"/workspace/{rel_path}"

            # Execute the script in the container
            exec_cmd = ["docker", "exec", self._container_name, "python", container_script_path]

            # Run the command with a timeout
            result = await asyncify(self._run_subprocess_with_timeout)(exec_cmd, timeout)

            return {
                "success": result[0],
                "stdout": result[1],
                "stderr": result[2],
                "returncode": result[3] if result[0] else 1,
            }

        except Exception as e:
            return {"success": False, "error": f"Execution error: {str(e)}"}

    def _run_subprocess_with_timeout(self, cmd: list[str], timeout: int) -> Tuple[bool, str, str, int]:
        """
        Run a subprocess with timeout and return status, stdout, stderr, and return code.

        Args:
            cmd: Command to run as a list of strings
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (success, stdout, stderr, return_code)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (result.returncode == 0, result.stdout, result.stderr, result.returncode)
        except subprocess.TimeoutExpired:
            return (False, "", f"Execution timed out after {timeout} seconds", -1)
        except Exception as e:
            return (False, "", str(e), -1)
