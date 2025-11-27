# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import subprocess
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class CmdResult:
    """Result of executing a shell command."""

    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool


class ShellCallOutcome(BaseModel):
    """Outcome of shell command execution."""

    type: str = Field(..., description="Type of outcome: 'exit' or 'timeout'")
    exit_code: int | None = Field(None, description="Exit code if type is 'exit'")


class ShellCommandOutput(BaseModel):
    """Output from a single shell command execution."""

    stdout: str = Field(default="", description="Standard output from the command")
    stderr: str = Field(default="", description="Standard error from the command")
    outcome: ShellCallOutcome = Field(..., description="Outcome of the command execution")


class ShellCallOutput(BaseModel):
    """Shell call output payload for Responses API."""

    type: str = Field(default="shell_call_output", description="Type identifier")
    call_id: str = Field(..., description="Call ID matching the shell_call")
    max_output_length: int | None = Field(None, description="Maximum output length to truncate")
    output: list[ShellCommandOutput] = Field(..., description="List of command outputs")


class ShellExecutor:
    """Executor for shell commands with timeout support."""

    def __init__(self, default_timeout: float = 60.0):
        """Initialize the shell executor.

        Args:
            default_timeout: Default timeout in seconds for command execution.
        """
        self.default_timeout = default_timeout

    def run(self, cmd: str, timeout: float | None = None) -> CmdResult:
        """Execute a shell command and return the result.

        Args:
            cmd: Shell command to execute.
            timeout: Timeout in seconds. If None, uses default_timeout.

        Returns:
            CmdResult with stdout, stderr, exit_code, and timed_out flag.
        """
        t = timeout or self.default_timeout
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            out, err = p.communicate(timeout=t)
            return CmdResult(stdout=out or "", stderr=err or "", exit_code=p.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            p.kill()
            out, err = p.communicate()
            return CmdResult(stdout=out or "", stderr=err or "", exit_code=p.returncode, timed_out=True)

    def run_commands(
        self, commands: list[str], timeout_ms: int | None = None
    ) -> list[ShellCommandOutput]:
        """Execute multiple shell commands concurrently and return results.

        Args:
            commands: List of shell commands to execute.
            timeout_ms: Timeout in milliseconds for each command.

        Returns:
            List of ShellCommandOutput objects.
        """
        timeout = (timeout_ms / 1000.0) if timeout_ms is not None else None
        results = []

        for cmd in commands:
            result = self.run(cmd, timeout=timeout)
            if result.timed_out:
                outcome = ShellCallOutcome(type="timeout")
            else:
                outcome = ShellCallOutcome(type="exit", exit_code=result.exit_code)

            results.append(
                ShellCommandOutput(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    outcome=outcome,
                )
            )

        return results