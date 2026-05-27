"""CLI integration tests — verify command registration and basic behavior."""

from __future__ import annotations

from ag2_cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCLIBasics:
    def test_help_shows_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "install" in result.output
        assert "create" in result.output
        assert "run" in result.output
        assert "chat" in result.output
        assert "serve" in result.output
        assert "test" in result.output

    def test_version_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ag2-cli" in result.output

    def test_no_args_shows_banner(self) -> None:
        result = runner.invoke(app, [])
        assert result.exit_code == 0
        assert "ag2" in result.output.lower()


class TestCreateSubcommands:
    def test_create_help(self) -> None:
        result = runner.invoke(app, ["create", "--help"])
        assert result.exit_code == 0
        assert "project" in result.output
        assert "agent" in result.output
        assert "tool" in result.output
        assert "team" in result.output

    def test_create_project_help(self) -> None:
        result = runner.invoke(app, ["create", "project", "--help"])
        assert result.exit_code == 0
        assert "template" in result.output.lower()


class TestInstallSubcommands:
    def test_install_help(self) -> None:
        result = runner.invoke(app, ["install", "--help"])
        assert result.exit_code == 0
        assert "skills" in result.output
        assert "list" in result.output
        assert "uninstall" in result.output

    def test_install_list_targets(self) -> None:
        result = runner.invoke(app, ["install", "list", "targets"])
        assert result.exit_code == 0
        assert "Cursor" in result.output or "cursor" in result.output


class TestTestSubcommands:
    def test_test_help(self) -> None:
        result = runner.invoke(app, ["test", "--help"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "bench" in result.output

    def test_bench_coming_soon(self) -> None:
        result = runner.invoke(app, ["test", "bench", "fake.py", "--suite", "gaia"])
        assert result.exit_code == 0
        assert "coming soon" in result.output


class TestRunCommand:
    def test_run_missing_file(self) -> None:
        result = runner.invoke(app, ["run", "nonexistent.py"])
        assert result.exit_code != 0

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "message" in result.output.lower()


class TestChatCommand:
    def test_chat_no_args_shows_error(self) -> None:
        result = runner.invoke(app, ["chat"])
        assert result.exit_code != 0

    def test_chat_help(self) -> None:
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower()


class TestServeCommand:
    def test_serve_help(self) -> None:
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()
        assert "protocol" in result.output.lower()

    def test_serve_mcp_requires_file(self) -> None:
        result = runner.invoke(app, ["serve", "fake.py", "--protocol", "mcp"])
        # MCP is now implemented but requires a valid file and dependencies
        assert result.exit_code == 1
