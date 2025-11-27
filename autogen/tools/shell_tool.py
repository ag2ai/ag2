# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
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
    """Executor for shell commands with timeout and sandboxing support.

    Provides multiple layers of security:
    1. Command pattern filtering (blocks dangerous commands)
    2. Working directory restriction (chroot-like behavior)
    3. Allowed/denied command lists (whitelist/blacklist)
    4. Path restrictions (limits file system access)
    """

    # Dangerous command patterns to block by default
    DEFAULT_DANGEROUS_PATTERNS = [
        (r"\brm\s+-rf\s+/", "Use of 'rm -rf /' is not allowed."),
        (r"\brm\s+-rf\s+~\b", "Use of 'rm -rf ~' is not allowed."),
        (r">\s*/dev/sd[a-z][1-9]?", "Overwriting disk blocks directly is not allowed."),
        (r":\(\)\{\s*:\|\:&\s*\};:", "Fork bombs are not allowed."),
        (r"\bdd\b.*if=.*of=", "Use of 'dd' with disk operations is not allowed."),
        (r"\bmkfs\.", "Formatting file systems is not allowed."),
        (r"\bformat\s+[A-Z]:", "Formatting drives is not allowed (Windows)."),
        (r"\bdel\s+/[sS]\s+C:\\", "System directory deletion is not allowed (Windows)."),
    ]

    def __init__(
        self,
        default_timeout: float = 60.0,
        workspace_dir: str | Path | None = None,
        allowed_paths: list[str] | None = None,
        allowed_commands: list[str] | None = None,
        denied_commands: list[str] | None = None,
        enable_command_filtering: bool = True,
        dangerous_patterns: list[tuple[str, str]] | None = None,
    ):
        """Initialize the shell executor with sandboxing options.

        Args:
            default_timeout: Default timeout in seconds for command execution.
            workspace_dir: Working directory for command execution. Commands will be executed
                     within this directory and cannot access files outside of it if path
                     restrictions are enabled. Defaults to current directory.
            allowed_paths: List of allowed path patterns (glob patterns). Commands can only
                          access paths matching these patterns. If None, all paths within
                          workspace_dir are allowed. Use ["**"] to allow all paths.
            allowed_commands: Whitelist of allowed commands (e.g., ["ls", "cat", "grep"]).
                            If provided, only these commands can be executed. None = allow all.
            denied_commands: Blacklist of denied commands (e.g., ["rm", "dd"]).
                           These commands will be blocked. None = use default dangerous patterns.
            enable_command_filtering: Whether to enable pattern-based command filtering.
                                    Defaults to True.
            dangerous_patterns: Custom dangerous command patterns to check. Each pattern is
                              a tuple of (regex_pattern, error_message). If None, uses
                              DEFAULT_DANGEROUS_PATTERNS.
        """
        self.default_timeout = default_timeout
        
        # Working directory setup
        if workspace_dir is None:
            self.workspace_dir = Path.cwd()
        else:
            self.workspace_dir = Path(workspace_dir).resolve()
            if not self.workspace_dir.exists():
                self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Path restrictions
        if allowed_paths is None:
            allowed_paths = ["**"]  # Allow all paths within workspace_dir by default
        self.allowed_paths = allowed_paths
        
        # Command restrictions
        self.allowed_commands = allowed_commands
        self.denied_commands = denied_commands or []
        
        # Pattern filtering
        self.enable_command_filtering = enable_command_filtering
        self.dangerous_patterns = dangerous_patterns or self.DEFAULT_DANGEROUS_PATTERNS

    def _validate_path(self, path: str | Path) -> bool:
        """Check if a path is allowed based on allowed_paths patterns.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is allowed, False otherwise
        """
        if "**" in self.allowed_paths:
            return True  # Allow all paths
        
        path_obj = Path(path)
        
        # Resolve relative to workspace_dir
        if not path_obj.is_absolute():
            path_obj = (self.workspace_dir / path_obj).resolve()
        
        # Check if path is within workspace_dir
        try:
            path_obj.relative_to(self.workspace_dir)
        except ValueError:
            return False  # Path is outside workspace_dir
        
        # Check against allowed patterns
        for pattern in self.allowed_paths:
            if path_obj.match(pattern) or self.workspace_dir.joinpath(pattern).match(str(path_obj)):
                return True
        
        return False

    def _validate_command(self, cmd: str) -> None:
        """Validate a command against security restrictions.
        
        Args:
            cmd: Command string to validate
            
        Raises:
            ValueError: If command violates security restrictions
        """
        # Extract command name (first word)
        cmd_parts = cmd.strip().split()
        if not cmd_parts:
            raise ValueError("Empty command is not allowed")
        
        cmd_name = cmd_parts[0].split("/")[-1]  # Get basename in case of /usr/bin/command
        
        # Check whitelist
        if self.allowed_commands is not None:
            if cmd_name not in self.allowed_commands:
                raise ValueError(f"Command '{cmd_name}' is not in the allowed commands list: {self.allowed_commands}")
        
        # Check blacklist
        if cmd_name in self.denied_commands:
            raise ValueError(f"Command '{cmd_name}' is in the denied commands list")
        
        # Check dangerous patterns
        if self.enable_command_filtering:
            for pattern, message in self.dangerous_patterns:
                if re.search(pattern, cmd, re.IGNORECASE):
                    raise ValueError(f"Potentially dangerous command detected: {message}")
        
        # Check for path access violations
        if self.allowed_paths and "**" not in self.allowed_paths:
            # Extract paths from command (simple heuristic - look for absolute paths or paths with /)
            path_pattern = r'(?:^|\s)([/~]|\.\./)[^\s]*'
            matches = re.findall(path_pattern, cmd)
            for match in matches:
                # Normalize path
                if match.startswith("~"):
                    test_path = os.path.expanduser(match)
                else:
                    test_path = match
                
                if not self._validate_path(test_path):
                    raise ValueError(f"Access to path '{match}' is not allowed. Allowed paths: {self.allowed_paths}")

    def run(self, cmd: str, timeout: float | None = None) -> CmdResult:
        """Execute a shell command and return the result.

        Args:
            cmd: Shell command to execute.
            timeout: Timeout in seconds. If None, uses default_timeout.

        Returns:
            CmdResult with stdout, stderr, exit_code, and timed_out flag.
            
        Raises:
            ValueError: If command violates security restrictions
        """
        # Validate command before execution
        self._validate_command(cmd)
        
        t = timeout or self.default_timeout
        
        # Execute in the restricted working directory
        p = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.workspace_dir),
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
            try:
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
            except ValueError as e:
                # Security violation - return error as output
                results.append(
                    ShellCommandOutput(
                        stdout="",
                        stderr=str(e),
                        outcome=ShellCallOutcome(type="exit", exit_code=1),
                    )
                )

        return results