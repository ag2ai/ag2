# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from autogen.tools.experimental.shell.shell_tool import CmdResult, ShellExecutor

# -----------------------------------------------------------------------------
# CmdResult Dataclass Tests
# -----------------------------------------------------------------------------


class TestCmdResult:
    """Test CmdResult dataclass."""

    def test_cmd_result_initialization(self) -> None:
        """Test CmdResult can be initialized with all fields."""
        result = CmdResult(
            stdout="Hello world",
            stderr="Error message",
            exit_code=0,
            timed_out=False,
        )

        assert result.stdout == "Hello world"
        assert result.stderr == "Error message"
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_cmd_result_with_none_exit_code(self) -> None:
        """Test CmdResult with None exit_code."""
        result = CmdResult(
            stdout="Output",
            stderr="",
            exit_code=None,
            timed_out=True,
        )

        assert result.exit_code is None
        assert result.timed_out is True

    def test_cmd_result_empty_strings(self) -> None:
        """Test CmdResult with empty strings."""
        result = CmdResult(stdout="", stderr="", exit_code=1, timed_out=False)

        assert result.stdout == ""
        assert result.stderr == ""
        assert result.exit_code == 1


# -----------------------------------------------------------------------------
# ShellExecutor Initialization Tests
# -----------------------------------------------------------------------------


class TestShellExecutorInit:
    """Test ShellExecutor initialization with various parameters."""

    def test_init_with_defaults(self) -> None:
        """Test ShellExecutor initialization with default parameters."""
        executor = ShellExecutor()

        assert executor.default_timeout == 60.0
        assert executor.workspace_dir == Path.cwd()
        assert executor.allowed_paths == ["**"]
        assert executor.allowed_commands is None
        assert executor.denied_commands == []
        assert executor.enable_command_filtering is True
        assert executor.dangerous_patterns == ShellExecutor.DEFAULT_DANGEROUS_PATTERNS

    def test_init_with_custom_timeout(self) -> None:
        """Test ShellExecutor with custom timeout."""
        executor = ShellExecutor(default_timeout=30.0)

        assert executor.default_timeout == 30.0

    def test_init_with_workspace_dir_string(self) -> None:
        """Test ShellExecutor with workspace_dir as string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir)

            assert executor.workspace_dir == Path(tmpdir).resolve()

    def test_init_with_workspace_dir_path(self) -> None:
        """Test ShellExecutor with workspace_dir as Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=Path(tmpdir))

            assert executor.workspace_dir == Path(tmpdir).resolve()

    def test_init_with_nonexistent_workspace_dir(self) -> None:
        """Test ShellExecutor creates nonexistent workspace_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_workspace"
            executor = ShellExecutor(workspace_dir=str(new_dir))

            assert executor.workspace_dir == new_dir.resolve()
            assert new_dir.exists()

    def test_init_with_allowed_paths(self) -> None:
        """Test ShellExecutor with custom allowed_paths."""
        executor = ShellExecutor(allowed_paths=["src/**", "tests/**"])

        assert executor.allowed_paths == ["src/**", "tests/**"]

    def test_init_with_none_allowed_paths(self) -> None:
        """Test ShellExecutor with None allowed_paths defaults to ['**']."""
        executor = ShellExecutor(allowed_paths=None)

        assert executor.allowed_paths == ["**"]

    def test_init_with_allowed_commands(self) -> None:
        """Test ShellExecutor with allowed_commands whitelist."""
        executor = ShellExecutor(allowed_commands=["ls", "cat", "grep"])

        assert executor.allowed_commands == ["ls", "cat", "grep"]

    def test_init_with_denied_commands(self) -> None:
        """Test ShellExecutor with denied_commands blacklist."""
        executor = ShellExecutor(denied_commands=["rm", "dd", "format"])

        assert executor.denied_commands == ["rm", "dd", "format"]

    def test_init_with_none_denied_commands(self) -> None:
        """Test ShellExecutor with None denied_commands defaults to empty list."""
        executor = ShellExecutor(denied_commands=None)

        assert executor.denied_commands == []

    def test_init_with_disable_command_filtering(self) -> None:
        """Test ShellExecutor with command filtering disabled."""
        executor = ShellExecutor(enable_command_filtering=False)

        assert executor.enable_command_filtering is False

    def test_init_with_custom_dangerous_patterns(self) -> None:
        """Test ShellExecutor with custom dangerous patterns."""
        custom_patterns = [
            (r"\brm\s+-rf", "Custom dangerous pattern"),
        ]
        executor = ShellExecutor(dangerous_patterns=custom_patterns)

        assert executor.dangerous_patterns == custom_patterns

    def test_init_with_none_dangerous_patterns(self) -> None:
        """Test ShellExecutor with None dangerous_patterns uses defaults."""
        executor = ShellExecutor(dangerous_patterns=None)

        assert executor.dangerous_patterns == ShellExecutor.DEFAULT_DANGEROUS_PATTERNS

    def test_init_with_all_parameters(self) -> None:
        """Test ShellExecutor with all parameters set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_patterns = [(r"\btest\b", "Test pattern")]
            executor = ShellExecutor(
                default_timeout=45.0,
                workspace_dir=tmpdir,
                allowed_paths=["src/**"],
                allowed_commands=["ls"],
                denied_commands=["rm"],
                enable_command_filtering=True,
                dangerous_patterns=custom_patterns,
            )

            assert executor.default_timeout == 45.0
            assert executor.workspace_dir == Path(tmpdir).resolve()
            assert executor.allowed_paths == ["src/**"]
            assert executor.allowed_commands == ["ls"]
            assert executor.denied_commands == ["rm"]
            assert executor.enable_command_filtering is True
            assert executor.dangerous_patterns == custom_patterns


# -----------------------------------------------------------------------------
# _validate_path Tests
# -----------------------------------------------------------------------------


class TestShellExecutorValidatePath:
    """Test ShellExecutor._validate_path method."""

    def test_validate_path_with_wildcard_allows_all(self) -> None:
        """Test _validate_path returns True when allowed_paths contains '**'."""
        executor = ShellExecutor(allowed_paths=["**"])

        assert executor._validate_path("/any/path") is True
        assert executor._validate_path("relative/path") is True
        assert executor._validate_path("/etc/passwd") is True

    def test_validate_path_within_workspace(self) -> None:
        """Test _validate_path allows paths within workspace_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["**"])

            workspace_file = Path(tmpdir) / "test.txt"
            workspace_file.touch()

            assert executor._validate_path(str(workspace_file)) is True
            assert executor._validate_path("test.txt") is True  # Relative path

    def test_validate_path_outside_workspace(self) -> None:
        """Test _validate_path rejects paths outside workspace_dir."""
        with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as other_dir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["src/**"])

            other_file = Path(other_dir) / "test.txt"
            other_file.touch()

            assert executor._validate_path(str(other_file)) is False

    def test_validate_path_with_pattern_matching(self) -> None:
        """Test _validate_path with pattern matching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["src/**", "tests/*.py"])

            src_file = Path(tmpdir) / "src" / "file.py"
            src_file.parent.mkdir(parents=True, exist_ok=True)
            src_file.touch()

            test_file = Path(tmpdir) / "tests" / "test.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.touch()

            # Use relative paths or absolute paths within workspace
            assert executor._validate_path(str(src_file.relative_to(tmpdir))) is True
            assert executor._validate_path(str(test_file.relative_to(tmpdir))) is True

            other_file = Path(tmpdir) / "other" / "file.py"
            other_file.parent.mkdir(parents=True, exist_ok=True)
            other_file.touch()

            assert executor._validate_path(str(other_file.relative_to(tmpdir))) is False

    def test_validate_path_absolute_path(self) -> None:
        """Test _validate_path with absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["**"])

            abs_path = Path(tmpdir).resolve() / "test.txt"
            abs_path.touch()

            assert executor._validate_path(str(abs_path)) is True

    def test_validate_path_relative_path(self) -> None:
        """Test _validate_path resolves relative paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["**"])

            # Relative path should be resolved relative to workspace_dir
            assert executor._validate_path("test.txt") is True


# -----------------------------------------------------------------------------
# _validate_command Tests
# -----------------------------------------------------------------------------


class TestShellExecutorValidateCommand:
    """Test ShellExecutor._validate_command method."""

    def test_validate_command_empty_command(self) -> None:
        """Test _validate_command raises error for empty command."""
        executor = ShellExecutor()

        with pytest.raises(ValueError, match="Empty command is not allowed"):
            executor._validate_command("")

        with pytest.raises(ValueError, match="Empty command is not allowed"):
            executor._validate_command("   ")

    def test_validate_command_with_allowed_commands_whitelist(self) -> None:
        """Test _validate_command with allowed_commands whitelist."""
        executor = ShellExecutor(allowed_commands=["ls", "cat", "echo"])

        # Allowed commands should pass
        executor._validate_command("ls -la")
        executor._validate_command("cat file.txt")
        executor._validate_command("echo hello")

        # Disallowed commands should fail
        with pytest.raises(ValueError, match="not in the allowed commands list"):
            executor._validate_command("rm file.txt")

        with pytest.raises(ValueError, match="not in the allowed commands list"):
            executor._validate_command("grep pattern file.txt")

    def test_validate_command_with_denied_commands_blacklist(self) -> None:
        """Test _validate_command with denied_commands blacklist."""
        executor = ShellExecutor(denied_commands=["rm", "dd", "format"])

        # Denied commands should fail
        with pytest.raises(ValueError, match="is in the denied commands list"):
            executor._validate_command("rm file.txt")

        with pytest.raises(ValueError, match="is in the denied commands list"):
            executor._validate_command("dd if=/dev/zero of=file")

        # Other commands should pass
        executor._validate_command("ls -la")
        executor._validate_command("cat file.txt")

    def test_validate_command_with_full_path(self) -> None:
        """Test _validate_command extracts command name from full path."""
        executor = ShellExecutor(allowed_commands=["ls"])

        # Should extract 'ls' from /usr/bin/ls
        executor._validate_command("/usr/bin/ls -la")

        executor = ShellExecutor(denied_commands=["rm"])

        with pytest.raises(ValueError, match="is in the denied commands list"):
            executor._validate_command("/bin/rm file.txt")

    def test_validate_command_dangerous_patterns(self) -> None:
        """Test _validate_command blocks dangerous patterns."""
        executor = ShellExecutor(enable_command_filtering=True)

        # Test various dangerous patterns
        dangerous_commands = [
            "rm -rf /",
            "rm -rf ~",
            "dd if=/dev/zero of=/dev/sda",
            "mkfs.ext4 /dev/sda1",
            "format C:",
        ]

        for cmd in dangerous_commands:
            with pytest.raises(ValueError, match="Potentially dangerous command detected"):
                executor._validate_command(cmd)

    def test_validate_command_with_filtering_disabled(self) -> None:
        """Test _validate_command skips pattern checking when filtering disabled."""
        executor = ShellExecutor(enable_command_filtering=False)

        # Dangerous commands should pass when filtering is disabled
        # (but may still be blocked by other checks)
        executor._validate_command("rm -rf /tmp/test")  # Should pass without pattern check

    def test_validate_command_path_validation(self) -> None:
        """Test _validate_command validates paths in commands."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(
                workspace_dir=tmpdir,
                allowed_paths=["src/**"],  # Not "**", so path validation is active
            )

            # Path within allowed pattern should pass
            src_file = Path(tmpdir) / "src" / "file.txt"
            src_file.parent.mkdir(parents=True, exist_ok=True)
            src_file.touch()

            # Use relative path to avoid regex matching issues
            executor._validate_command("cat src/file.txt")

            # Path outside allowed pattern should fail
            # Use absolute path or path starting with / to trigger regex matching
            other_file = Path(tmpdir) / "other" / "file.txt"
            other_file.parent.mkdir(parents=True, exist_ok=True)
            other_file.touch()

            # Use absolute path to trigger the regex pattern matching
            with pytest.raises(ValueError, match="Access to path.*is not allowed"):
                executor._validate_command(f"cat {other_file}")

            # Or use a path with ../ to trigger validation
            with pytest.raises(ValueError, match="Access to path.*is not allowed"):
                executor._validate_command("cat ../other/file.txt")

    def test_validate_command_path_validation_with_wildcard(self) -> None:
        """Test _validate_command skips path validation when allowed_paths has '**'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["**"])

            # All paths should be allowed
            executor._validate_command("cat /any/path/file.txt")
            executor._validate_command("rm /etc/passwd")  # Path validation skipped

    def test_validate_command_with_tilde_path(self) -> None:
        """Test _validate_command handles tilde paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir, allowed_paths=["**"])

            # Tilde paths should be expanded
            executor._validate_command("cat ~/file.txt")

    def test_validate_command_path_validation_windows_style(self) -> None:
        """Test _validate_command validates Windows-style paths (e.g., C:\\path)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(
                workspace_dir=tmpdir,
                allowed_paths=["src/**"],  # Not "**", so path validation is active
            )

            # Windows-style absolute paths should be detected and validated
            # These paths are outside the allowed patterns and should fail
            with pytest.raises(ValueError, match="Access to path.*is not allowed"):
                executor._validate_command("cat C:\\Windows\\System32\\file.txt")

            with pytest.raises(ValueError, match="Access to path.*is not allowed"):
                executor._validate_command("type D:\\Users\\test\\file.txt")


# -----------------------------------------------------------------------------
# run Method Tests
# -----------------------------------------------------------------------------


class TestShellExecutorRun:
    """Test ShellExecutor.run method."""

    def test_run_simple_command(self) -> None:
        """Test run executes a simple command successfully."""
        executor = ShellExecutor()

        result = executor.run("echo hello")

        assert result.stdout.strip() == "hello"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.timed_out is False

    def test_run_command_with_stderr(self) -> None:
        """Test run captures stderr output."""
        executor = ShellExecutor()

        # Use a command that produces stderr (command not found on most systems)
        result = executor.run("echo hello 1>&2")

        # On Unix, redirecting stdout to stderr should work
        assert result.stderr.strip() == "hello" or result.stdout.strip() == "hello"
        assert result.exit_code == 0

    def test_run_command_with_nonzero_exit(self) -> None:
        """Test run captures non-zero exit codes."""
        executor = ShellExecutor()

        # Use a command that fails
        result = executor.run("false")  # false always exits with 1

        assert result.exit_code == 1
        assert result.timed_out is False

    def test_run_with_custom_timeout(self) -> None:
        """Test run uses custom timeout parameter."""
        executor = ShellExecutor(default_timeout=60.0)

        # Short timeout for quick command
        result = executor.run("echo test", timeout=1.0)

        assert result.stdout.strip() == "test"
        assert result.timed_out is False

    def test_run_with_default_timeout(self) -> None:
        """Test run uses default_timeout when timeout not specified."""
        executor = ShellExecutor(default_timeout=30.0)

        result = executor.run("echo test")

        assert result.stdout.strip() == "test"
        assert result.timed_out is False

    def test_run_in_workspace_directory(self) -> None:
        """Test run executes commands in workspace_dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            executor = ShellExecutor(workspace_dir=tmpdir)

            # Create a file in workspace
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            # Command should see the file
            result = executor.run("cat test.txt")

            assert "test content" in result.stdout
            assert result.exit_code == 0

    def test_run_blocks_dangerous_command(self) -> None:
        """Test run blocks dangerous commands."""
        executor = ShellExecutor()

        with pytest.raises(ValueError, match="Potentially dangerous command detected"):
            executor.run("rm -rf /")

    def test_run_blocks_denied_command(self) -> None:
        """Test run blocks denied commands."""
        executor = ShellExecutor(denied_commands=["rm"])

        with pytest.raises(ValueError, match="is in the denied commands list"):
            executor.run("rm file.txt")

    def test_run_blocks_command_not_in_whitelist(self) -> None:
        """Test run blocks commands not in allowed_commands."""
        executor = ShellExecutor(allowed_commands=["ls", "cat"])

        with pytest.raises(ValueError, match="not in the allowed commands list"):
            executor.run("rm file.txt")

    def test_run_allows_command_in_whitelist(self) -> None:
        """Test run allows commands in allowed_commands."""
        executor = ShellExecutor(allowed_commands=["echo", "ls"])

        result = executor.run("echo test")

        assert result.stdout.strip() == "test"
        assert result.exit_code == 0


# -----------------------------------------------------------------------------
# run_commands Method Tests
# -----------------------------------------------------------------------------


class TestShellExecutorRunCommands:
    """Test ShellExecutor.run_commands method."""

    def test_run_commands_single_command(self) -> None:
        """Test run_commands with a single command."""
        executor = ShellExecutor()

        results = executor.run_commands(["echo hello"])

        assert len(results) == 1
        assert results[0].stdout.strip() == "hello"
        assert results[0].stderr == ""
        assert results[0].outcome.type == "exit"
        assert results[0].outcome.exit_code == 0

    def test_run_commands_multiple_commands(self) -> None:
        """Test run_commands with multiple commands."""
        executor = ShellExecutor()

        results = executor.run_commands(["echo first", "echo second", "echo third"])

        assert len(results) == 3
        assert results[0].stdout.strip() == "first"
        assert results[1].stdout.strip() == "second"
        assert results[2].stdout.strip() == "third"
        assert all(r.outcome.type == "exit" and r.outcome.exit_code == 0 for r in results)

    def test_run_commands_with_timeout_ms(self) -> None:
        """Test run_commands with timeout_ms parameter."""
        executor = ShellExecutor()

        results = executor.run_commands(["echo test"], timeout_ms=5000)

        assert len(results) == 1
        assert results[0].stdout.strip() == "test"
        assert results[0].outcome.type == "exit"

    def test_run_commands_with_failing_command(self) -> None:
        """Test run_commands handles commands that fail."""
        executor = ShellExecutor()

        results = executor.run_commands(["false"])  # false always exits with 1

        assert len(results) == 1
        assert results[0].outcome.type == "exit"
        assert results[0].outcome.exit_code == 1

    def test_run_commands_mixed_success_and_failure(self) -> None:
        """Test run_commands with mix of successful and failing commands."""
        executor = ShellExecutor()

        results = executor.run_commands(["echo success", "false", "echo after"])

        assert len(results) == 3
        assert results[0].stdout.strip() == "success"
        assert results[0].outcome.exit_code == 0
        assert results[1].outcome.exit_code == 1
        assert results[2].stdout.strip() == "after"
        assert results[2].outcome.exit_code == 0

    def test_run_commands_handles_security_violation(self) -> None:
        """Test run_commands handles security violations gracefully."""
        executor = ShellExecutor(denied_commands=["rm"])

        results = executor.run_commands(["echo test", "rm file.txt", "echo after"])

        assert len(results) == 3
        # First command succeeds
        assert results[0].stdout.strip() == "test"
        assert results[0].outcome.exit_code == 0
        # Second command is blocked
        assert "is in the denied commands list" in results[1].stderr
        assert results[1].outcome.type == "exit"
        assert results[1].outcome.exit_code == 1
        # Third command still executes
        assert results[2].stdout.strip() == "after"
        assert results[2].outcome.exit_code == 0

    def test_run_commands_with_dangerous_pattern(self) -> None:
        """Test run_commands blocks dangerous patterns."""
        executor = ShellExecutor()

        results = executor.run_commands(["echo safe", "rm -rf /", "echo after"])

        assert len(results) == 3
        assert results[0].stdout.strip() == "safe"
        assert "Potentially dangerous command detected" in results[1].stderr
        assert results[1].outcome.exit_code == 1
        assert results[2].stdout.strip() == "after"

    def test_run_commands_empty_list(self) -> None:
        """Test run_commands with empty command list."""
        executor = ShellExecutor()

        results = executor.run_commands([])

        assert len(results) == 0

    def test_run_commands_with_none_timeout(self) -> None:
        """Test run_commands with None timeout_ms."""
        executor = ShellExecutor(default_timeout=10.0)

        results = executor.run_commands(["echo test"], timeout_ms=None)

        assert len(results) == 1
        assert results[0].stdout.strip() == "test"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestShellExecutorIntegration:
    """Integration tests for ShellExecutor."""

    def test_full_workflow_with_restrictions(self) -> None:
        """Test complete workflow with all security restrictions enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = Path(tmpdir) / "src"
            src_dir.mkdir()
            test_file = src_dir / "test.txt"
            test_file.write_text("test content")

            executor = ShellExecutor(
                workspace_dir=tmpdir,
                allowed_paths=["src/**"],
                allowed_commands=["cat", "ls"],
                denied_commands=["rm"],
                enable_command_filtering=True,
            )

            # Allowed command with allowed path
            results = executor.run_commands(["cat src/test.txt"])

            assert len(results) == 1
            assert "test content" in results[0].stdout

            # Denied command should be blocked, but whitelist check happens first
            # So we expect whitelist error, not blacklist error
            results = executor.run_commands(["rm src/test.txt"])

            # When both whitelist and blacklist are set, whitelist is checked first
            assert "not in the allowed commands list" in results[0].stderr

            # Test blacklist when whitelist is not set
            executor_no_whitelist = ShellExecutor(
                workspace_dir=tmpdir,
                allowed_paths=["src/**"],
                allowed_commands=None,  # No whitelist
                denied_commands=["rm"],
                enable_command_filtering=True,
            )

            results = executor_no_whitelist.run_commands(["rm src/test.txt"])

            assert "is in the denied commands list" in results[0].stderr

            # Command not in whitelist should be blocked
            results = executor.run_commands(["grep pattern src/test.txt"])

            assert "not in the allowed commands list" in results[0].stderr

    def test_default_dangerous_patterns_are_defined(self) -> None:
        """Test that DEFAULT_DANGEROUS_PATTERNS contains expected patterns."""
        import re

        patterns = ShellExecutor.DEFAULT_DANGEROUS_PATTERNS

        assert len(patterns) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in patterns)
        assert all(isinstance(p[0], str) and isinstance(p[1], str) for p in patterns)

        # Check for some expected dangerous patterns by testing if they match
        pattern_strings = [p[0] for p in patterns]

        # Test if patterns match dangerous commands (they are regex patterns)
        rm_pattern_found = any(re.search(p, "rm -rf /", re.IGNORECASE) for p in pattern_strings)
        dd_pattern_found = any(re.search(p, "dd if=/dev/zero of=/dev/sda", re.IGNORECASE) for p in pattern_strings)

        assert rm_pattern_found, "No pattern found that matches 'rm -rf /'"
        assert dd_pattern_found, "No pattern found that matches 'dd of=/dev/sda'"
