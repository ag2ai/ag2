"""Tests for new ag2 create features — from-description, from-openapi, from-module, scaffold fixes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ag2_cli.app import app
from ag2_cli.commands.create import (
    _detect_generation_model,
    _parse_json_response,
)
from typer.testing import CliRunner

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_generate(response_json: dict) -> MagicMock:
    """Create a mock for _llm_generate that returns a JSON string."""
    return MagicMock(return_value=json.dumps(response_json))


# ---------------------------------------------------------------------------
# TestDetectGenerationModel
# ---------------------------------------------------------------------------


class TestDetectGenerationModel:
    def test_detects_openai(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False):  # pragma: allowlist secret
            model = _detect_generation_model()
        assert model == "gpt-4o"

    def test_detects_anthropic(self) -> None:
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}  # pragma: allowlist secret
        with patch.dict(os.environ, env, clear=True):
            model = _detect_generation_model()
        assert model is not None
        assert "claude" in model

    def test_detects_google(self) -> None:
        env = {"GOOGLE_API_KEY": "AIza-test"}  # pragma: allowlist secret
        with patch.dict(os.environ, env, clear=True):
            model = _detect_generation_model()
        assert model is not None
        assert "gemini" in model

    def test_returns_none_when_no_keys(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            model = _detect_generation_model()
        assert model is None


# ---------------------------------------------------------------------------
# TestParseJsonResponse
# ---------------------------------------------------------------------------


class TestParseJsonResponse:
    def test_parses_plain_json(self) -> None:
        text = '{"name": "test", "value": 42}'
        result = _parse_json_response(text)
        assert result == {"name": "test", "value": 42}

    def test_parses_json_in_markdown_block(self) -> None:
        text = 'Here is the result:\n```json\n{"name": "test"}\n```\nDone.'
        result = _parse_json_response(text)
        assert result == {"name": "test"}

    def test_parses_json_in_generic_code_block(self) -> None:
        text = 'Result:\n```\n{"key": "val"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "val"}

    def test_extracts_json_object_from_text(self) -> None:
        text = 'Some preamble text {"name": "extracted"} and more text'
        result = _parse_json_response(text)
        assert result == {"name": "extracted"}

    def test_exits_on_unparsable_response(self) -> None:
        with pytest.raises((SystemExit, Exception)):
            _parse_json_response("This is not JSON at all")


# ---------------------------------------------------------------------------
# TestFullstackAgenticTemplate
# ---------------------------------------------------------------------------


class TestFullstackAgenticTemplate:
    def test_creates_all_agents(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "fa-test", "--template", "fullstack-agentic"])
        assert result.exit_code == 0
        project = tmp_path / "fa-test"
        assert (project / "agents" / "planner.py").is_file()
        assert (project / "agents" / "coder.py").is_file()
        assert (project / "agents" / "reviewer.py").is_file()

    def test_agent_files_contain_valid_code(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "fa-test", "--template", "fullstack-agentic"])
        content = (tmp_path / "fa-test" / "agents" / "planner.py").read_text()
        assert "AssistantAgent" in content
        assert 'name="planner"' in content
        assert 'LLMConfig({"model": "gpt-4o"})' in content


# ---------------------------------------------------------------------------
# TestProjectScaffoldFixes
# ---------------------------------------------------------------------------


class TestProjectScaffoldFixes:
    """Verify config/llm.yaml and tools/web_search.py are created."""

    def test_config_llm_yaml_created(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "scaffold-test"])
        assert (tmp_path / "scaffold-test" / "config" / "llm.yaml").is_file()

    def test_tools_web_search_created(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "scaffold-test"])
        tool_file = tmp_path / "scaffold-test" / "tools" / "web_search.py"
        assert tool_file.is_file()
        content = tool_file.read_text()
        assert "@tool" in content
        assert "web_search" in content

    def test_all_templates_create_config_and_tools(self, tmp_path: Path) -> None:
        for template in ("blank", "research-team", "rag-chatbot", "fullstack-agentic"):
            os.chdir(tmp_path)
            name = f"proj-{template}"
            result = runner.invoke(app, ["create", "project", name, "--template", template])
            assert result.exit_code == 0, f"Template {template} failed: {result.output}"
            assert (tmp_path / name / "config" / "llm.yaml").is_file()
            assert (tmp_path / name / "tools" / "web_search.py").is_file()


# ---------------------------------------------------------------------------
# TestCreateProjectFromDescription
# ---------------------------------------------------------------------------


class TestCreateProjectFromDescription:
    _SAMPLE_SPEC = {
        "name": "slack-bot",
        "description": "A Slack monitoring bot",
        "agents": [
            {
                "name": "monitor",
                "system_message": "You monitor Slack channels.",
                "tools": ["slack_reader"],
            },
            {
                "name": "summarizer",
                "system_message": "You summarize discussions.",
                "tools": [],
            },
        ],
        "tools": [
            {
                "name": "slack_reader",
                "description": "Read Slack messages",
                "params": [{"name": "channel", "type": "str", "description": "Channel name"}],
            },
        ],
        "pattern": "auto",
    }

    @patch("ag2_cli.commands.create._llm_generate")
    def test_generates_project_from_description(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_SPEC)
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "--from-description", "A Slack bot that monitors channels"])
        assert result.exit_code == 0

        project = tmp_path / "slack-bot"
        assert project.is_dir()
        assert (project / "pyproject.toml").is_file()
        assert (project / "agents" / "monitor.py").is_file()
        assert (project / "agents" / "summarizer.py").is_file()
        assert (project / "tools" / "slack_reader.py").is_file()
        assert (project / "main.py").is_file()
        assert (project / "tests" / "test_agents.py").is_file()

    @patch("ag2_cli.commands.create._llm_generate")
    def test_name_override(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_SPEC)
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "my-custom-name", "--from-description", "A bot"])
        assert result.exit_code == 0
        assert (tmp_path / "my-custom-name").is_dir()

    @patch("ag2_cli.commands.create._llm_generate")
    def test_main_py_uses_group_chat_for_multiple_agents(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_SPEC)
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "--from-description", "multi-agent"])
        main_content = (tmp_path / "slack-bot" / "main.py").read_text()
        assert "run_group_chat" in main_content
        assert "AutoPattern" in main_content

    @patch("ag2_cli.commands.create._llm_generate")
    def test_single_agent_project_uses_initiate_chat(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        spec = {
            "name": "simple-bot",
            "agents": [{"name": "assistant", "system_message": "You help.", "tools": []}],
            "tools": [],
            "pattern": "auto",
        }
        mock_gen.return_value = json.dumps(spec)
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "--from-description", "simple bot"])
        main_content = (tmp_path / "simple-bot" / "main.py").read_text()
        assert "initiate_chat" in main_content
        assert "run_group_chat" not in main_content

    @patch("ag2_cli.commands.create._llm_generate")
    def test_fails_on_existing_dir(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_SPEC)
        os.chdir(tmp_path)
        (tmp_path / "slack-bot").mkdir()
        result = runner.invoke(app, ["create", "project", "--from-description", "A Slack bot"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    @patch("ag2_cli.commands.create._llm_generate")
    def test_fails_on_empty_agents(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        mock_gen.return_value = json.dumps({"name": "empty", "agents": [], "tools": []})
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project", "--from-description", "An empty project"])
        assert result.exit_code != 0
        assert "did not generate" in result.output.lower()

    def test_requires_name_without_from_description(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "project"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# TestCreateAgentFromDescription
# ---------------------------------------------------------------------------


class TestCreateAgentFromDescription:
    _SAMPLE_AGENT = {
        "name": "hn_watcher",
        "system_message": "You monitor Hacker News for AI papers.",
        "tools": [
            {
                "name": "hn_search",
                "description": "Search Hacker News",
                "params": [{"name": "query", "type": "str", "description": "Search query"}],
            },
        ],
    }

    @patch("ag2_cli.commands.create._llm_generate")
    def test_generates_agent_and_tools(self, mock_gen: MagicMock, tmp_project: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_AGENT)
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "agent", "--from-description", "Monitor HN for AI papers"])
        assert result.exit_code == 0
        assert (tmp_project / "agents" / "hn_watcher.py").is_file()
        assert (tmp_project / "tools" / "hn_search.py").is_file()

    @patch("ag2_cli.commands.create._llm_generate")
    def test_agent_file_has_tool_registration(self, mock_gen: MagicMock, tmp_project: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_AGENT)
        os.chdir(tmp_project)
        runner.invoke(app, ["create", "agent", "--from-description", "Monitor HN"])
        content = (tmp_project / "agents" / "hn_watcher.py").read_text()
        assert "from tools.hn_search import hn_search" in content
        assert "hn_search.register_tool(hn_watcher)" in content

    @patch("ag2_cli.commands.create._llm_generate")
    def test_name_override(self, mock_gen: MagicMock, tmp_project: Path) -> None:
        mock_gen.return_value = json.dumps(self._SAMPLE_AGENT)
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "agent", "custom-name", "--from-description", "Monitor HN"])
        assert result.exit_code == 0
        assert (tmp_project / "agents" / "custom_name.py").is_file()

    @patch("ag2_cli.commands.create._llm_generate")
    def test_agent_without_tools(self, mock_gen: MagicMock, tmp_project: Path) -> None:
        spec = {"name": "thinker", "system_message": "You think deeply.", "tools": []}
        mock_gen.return_value = json.dumps(spec)
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "agent", "--from-description", "A thinking agent"])
        assert result.exit_code == 0
        content = (tmp_project / "agents" / "thinker.py").read_text()
        assert "register_tool" not in content

    @patch("ag2_cli.commands.create._llm_generate")
    def test_creates_in_cwd_when_no_agents_dir(self, mock_gen: MagicMock, tmp_path: Path) -> None:
        spec = {"name": "solo", "system_message": "You are solo.", "tools": []}
        mock_gen.return_value = json.dumps(spec)
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "agent", "--from-description", "A solo agent"])
        assert result.exit_code == 0
        assert (tmp_path / "solo.py").is_file()

    def test_requires_name_without_from_description(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "agent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# TestCreateToolFromModule
# ---------------------------------------------------------------------------


class TestCreateToolFromModule:
    def test_generates_tools_from_json_module(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "tool", "--from-module", "json", "--functions", "dumps,loads"])
        assert result.exit_code == 0
        assert (tmp_path / "json_tools.py").is_file()
        content = (tmp_path / "json_tools.py").read_text()
        assert "json_dumps" in content
        assert "json_loads" in content

    def test_generates_with_custom_name(self, tmp_project: Path) -> None:
        os.chdir(tmp_project)
        result = runner.invoke(app, ["create", "tool", "my-json", "--from-module", "json", "--functions", "dumps"])
        assert result.exit_code == 0
        assert (tmp_project / "tools" / "my_json.py").is_file()

    def test_fails_on_nonexistent_module(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "tool", "--from-module", "nonexistent_module_xyz"])
        assert result.exit_code != 0

    def test_requires_name_without_from_flags(self, tmp_path: Path) -> None:
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "tool"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# TestCreateToolFromOpenAPI
# ---------------------------------------------------------------------------


class TestCreateToolFromOpenAPI:
    @patch("ag2_cli.commands.proxy._load_openapi_spec")
    @patch("ag2_cli.commands.proxy._parse_openapi_spec")
    @patch("ag2_cli.commands.proxy._generate_tool_file")
    def test_generates_tools_from_openapi(
        self,
        mock_gen: MagicMock,
        mock_parse: MagicMock,
        mock_load: MagicMock,
        tmp_path: Path,
    ) -> None:
        from ag2_cli.commands.proxy import ToolSpec

        mock_load.return_value = {"paths": {}}
        mock_parse.return_value = [
            ToolSpec(name="list_users", description="List users", source_type="openapi"),
            ToolSpec(name="get_user", description="Get user", source_type="openapi"),
        ]
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "tool", "--from-openapi", "https://example.com/openapi.json"])
        assert result.exit_code == 0
        assert "2 tools" in result.output
        mock_gen.assert_called_once()

    @patch("ag2_cli.commands.proxy._load_openapi_spec")
    def test_fails_on_bad_spec(self, mock_load: MagicMock, tmp_path: Path) -> None:
        mock_load.side_effect = Exception("Invalid spec")
        os.chdir(tmp_path)
        result = runner.invoke(app, ["create", "tool", "--from-openapi", "bad-url"])
        assert result.exit_code != 0
        assert "Failed to load" in result.output


# ---------------------------------------------------------------------------
# TestLLMConfigBugFix
# ---------------------------------------------------------------------------


class TestLLMConfigBugFix:
    """Verify the LLMConfig(api_type=...) → LLMConfig({...}) fix in all files."""

    def test_run_chat_uses_dict_config(self) -> None:
        """run.py chat_cmd should create LLMConfig with dict, not api_type kwarg."""
        import ast

        source = (Path(__file__).parent.parent / "src" / "ag2_cli" / "commands" / "run.py").read_text()
        tree = ast.parse(source)
        # Ensure no LLMConfig call has api_type keyword
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name == "LLMConfig":
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "api_type" not in kw_names, "LLMConfig still has api_type keyword in run.py"

    def test_discovery_uses_dict_config(self) -> None:
        """discovery.py should create LLMConfig with dict, not api_type kwarg."""
        import ast

        source = (Path(__file__).parent.parent / "src" / "ag2_cli" / "core" / "discovery.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name == "LLMConfig":
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "api_type" not in kw_names, "LLMConfig still has api_type keyword in discovery.py"

    def test_assertions_uses_dict_config(self) -> None:
        """assertions.py should create LLMConfig with dict, not api_type kwarg."""
        import ast

        source = (Path(__file__).parent.parent / "src" / "ag2_cli" / "testing" / "assertions.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = ""
                if isinstance(func, ast.Attribute):
                    name = func.attr
                elif isinstance(func, ast.Name):
                    name = func.id
                if name == "LLMConfig":
                    kw_names = [kw.arg for kw in node.keywords]
                    assert "api_type" not in kw_names, "LLMConfig still has api_type keyword in assertions.py"

    def test_scaffolded_agent_uses_dict_config(self, tmp_path: Path) -> None:
        """Generated agent files should use LLMConfig({...}) not LLMConfig(api_type=...)."""
        os.chdir(tmp_path)
        runner.invoke(app, ["create", "project", "cfg-test"])
        content = (tmp_path / "cfg-test" / "agents" / "assistant.py").read_text()
        assert 'LLMConfig({"model": "gpt-4o"})' in content
        assert "api_type" not in content


# ---------------------------------------------------------------------------
# TestTestCmdBaselineRemoved
# ---------------------------------------------------------------------------


class TestTestCmdChanges:
    """Verify the --baseline removal and --models coming-soon."""

    def test_baseline_param_removed(self) -> None:
        """test eval should not accept --baseline."""
        import sys
        from unittest.mock import patch as p

        with p.dict(sys.modules, {"autogen": MagicMock()}):
            result = runner.invoke(
                app,
                ["test", "eval", "fake.py", "--eval", "fake.yaml", "--baseline", "old.json"],
            )
        # Typer should reject unknown option
        assert result.exit_code != 0

    def test_models_shows_coming_soon(self, eval_yaml_file: Path, agent_file_with_main: Path) -> None:
        """--models should print coming soon warning."""
        import sys

        with (
            patch.dict(sys.modules, {"autogen": MagicMock()}),
            patch("ag2_cli.commands.test._run_single_case") as mock_run,
        ):
            from ag2_cli.testing import CaseResult
            from ag2_cli.testing.assertions import AssertionResult
            from ag2_cli.testing.cases import EvalAssertion, EvalCase

            case = EvalCase(name="basic_test", input="hello", assertions=[EvalAssertion(type="contains", value="x")])
            mock_run.return_value = CaseResult(
                case=case,
                assertion_results=[AssertionResult(passed=True, assertion_type="contains", message="ok")],
                output="x",
                turns=1,
                elapsed=0.1,
            )
            result = runner.invoke(
                app,
                [
                    "test",
                    "eval",
                    str(agent_file_with_main),
                    "--eval",
                    str(eval_yaml_file),
                    "--models",
                    "gpt-4o,claude-sonnet",
                ],
            )
        assert "coming soon" in result.output.lower()
