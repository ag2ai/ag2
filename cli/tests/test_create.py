"""Tests for ag2 create — scaffold generation."""

from __future__ import annotations

import os
from pathlib import Path

from ag2_cli.app import app
from typer.testing import CliRunner

runner = CliRunner()


class TestCreateProject:
    def test_creates_project_structure(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "my-bot"])
        assert result.exit_code == 0

        project = tmp_path / "my-bot"
        assert project.is_dir()
        assert (project / "pyproject.toml").is_file()
        assert (project / ".env.example").is_file()
        assert (project / ".gitignore").is_file()
        assert (project / "main.py").is_file()
        assert (project / "agents" / "__init__.py").is_file()
        assert (project / "agents" / "assistant.py").is_file()
        assert (project / "tools" / "__init__.py").is_file()
        assert (project / "tests" / "__init__.py").is_file()
        assert (project / "tests" / "test_agents.py").is_file()

    def test_pyproject_has_correct_name(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "my-app"])
        content = (tmp_path / "my-app" / "pyproject.toml").read_text()
        assert 'name = "my-app"' in content
        assert "ag2" in content

    def test_research_team_template(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "research", "--template", "research-team"])
        assert result.exit_code == 0
        project = tmp_path / "research"
        assert (project / "agents" / "researcher.py").is_file()
        assert (project / "agents" / "writer.py").is_file()

    def test_fails_on_existing_directory(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        (tmp_path / "existing").mkdir()
        result = runner.invoke(app, ["create", "project", "existing"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_fails_on_unknown_template(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "test", "--template", "nonexistent"])
        assert result.exit_code != 0
        assert "Unknown template" in result.output


class TestCreateAgent:
    def test_creates_agent_file(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "agent", "researcher"])
        assert result.exit_code == 0
        agent_file = tmp_project / "agents" / "researcher.py"
        assert agent_file.is_file()
        content = agent_file.read_text()
        assert "AssistantAgent" in content
        assert 'name="researcher"' in content

    def test_creates_agent_with_tools(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "agent", "helper", "--tools", "web-search,code-exec"])
        assert result.exit_code == 0
        content = (tmp_project / "agents" / "helper.py").read_text()
        assert "web_search" in content
        assert "code_exec" in content

    def test_fails_on_existing_file(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        (tmp_project / "agents" / "existing.py").write_text("# existing\n")
        result = runner.invoke(app, ["create", "agent", "existing"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_creates_in_cwd_when_no_agents_dir(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "agent", "my-agent"])
        assert result.exit_code == 0
        assert (tmp_path / "my_agent.py").is_file()


class TestCreateTool:
    def test_creates_tool_file(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "tool", "stock-price", "-d", "Fetch stock prices"])
        assert result.exit_code == 0
        tool_file = tmp_project / "tools" / "stock_price.py"
        assert tool_file.is_file()
        content = tool_file.read_text()
        assert "@tool" in content
        assert "stock_price" in content
        assert "Fetch stock prices" in content

    def test_creates_tool_without_description(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "tool", "my-tool"])
        assert result.exit_code == 0
        content = (tmp_project / "tools" / "my_tool.py").read_text()
        assert "@tool" in content


class TestCreateTeam:
    def test_creates_team_file(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(
            app,
            ["create", "team", "code-review", "--pattern", "round-robin", "--agents", "reviewer,tester,merger"],
        )
        assert result.exit_code == 0
        team_file = tmp_project / "teams" / "code_review.py"
        assert team_file.is_file()
        content = team_file.read_text()
        assert "RoundRobinPattern" in content
        assert "reviewer" in content
        assert "tester" in content
        assert "merger" in content

    def test_default_agents_when_none_specified(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "team", "my-team"])
        assert result.exit_code == 0
        team_file = tmp_project / "teams" / "my_team.py"
        content = team_file.read_text()
        assert "agent_a" in content
        assert "agent_b" in content

    def test_fails_on_unknown_pattern(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "team", "test", "--pattern", "invalid"])
        assert result.exit_code != 0
        assert "Unknown pattern" in result.output
