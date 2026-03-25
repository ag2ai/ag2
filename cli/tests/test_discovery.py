"""Tests for agent discovery from Python files and YAML configs."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from ag2_cli.core.discovery import (
    discover,
    import_agent_file,
    load_yaml_config,
)


class TestImportAgentFile:
    def test_imports_valid_python_file(self, agent_file_with_main: Path) -> None:
        module = import_agent_file(agent_file_with_main)
        assert hasattr(module, "main")
        assert callable(module.main)

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_agent_file(tmp_path / "nonexistent.py")

    def test_raises_on_non_python_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .py file"):
            import_agent_file(f)

    def test_module_can_import_sibling(self, tmp_path: Path) -> None:
        """Verify that the imported file can import siblings in its directory."""
        (tmp_path / "helper.py").write_text("VALUE = 42\n")
        (tmp_path / "my_agent.py").write_text(
            textwrap.dedent("""\
            from helper import VALUE
            result = VALUE
            """)
        )
        module = import_agent_file(tmp_path / "my_agent.py")
        assert module.result == 42


class TestDiscover:
    def test_discovers_main_function(self, agent_file_with_main: Path) -> None:
        d = discover(agent_file_with_main)
        assert d.kind == "main"
        assert d.main_fn is not None
        assert callable(d.main_fn)

    def test_discovers_agent_variable(self, agent_file_with_variable: Path) -> None:
        d = discover(agent_file_with_variable)
        assert d.kind == "agent"
        assert d.agent is not None
        assert d.agent.name == "researcher"
        assert d.agent_names == ["researcher"]

    def test_discovers_agents_list(self, agent_file_with_agents_list: Path) -> None:
        d = discover(agent_file_with_agents_list)
        assert d.kind == "agents"
        assert len(d.agents) == 2
        assert d.agent_names == ["alice", "bob"]

    def test_raises_on_empty_file(self, agent_file_empty: Path) -> None:
        with pytest.raises(ValueError, match="No runnable agent found"):
            discover(agent_file_empty)

    def test_discovers_team_variable(self, tmp_path: Path) -> None:
        f = tmp_path / "my_team.py"
        f.write_text(
            textwrap.dedent("""\
            class FakeAgent:
                def __init__(self, name):
                    self.name = name
                def initiate_chat(self, recipient, message=""):
                    pass

            team = FakeAgent("my-team")
            """)
        )
        d = discover(f)
        assert d.kind == "agent"
        assert d.agent.name == "my-team"

    def test_discovers_single_agent_instance(self, tmp_path: Path) -> None:
        """When no standard names are found, discover any agent-like object."""
        f = tmp_path / "my_agent.py"
        f.write_text(
            textwrap.dedent("""\
            class FakeAgent:
                def __init__(self, name):
                    self.name = name
                def initiate_chat(self, recipient, message=""):
                    pass

            my_custom_agent = FakeAgent("custom")
            """)
        )
        d = discover(f)
        assert d.kind == "agent"
        assert d.agent.name == "custom"

    def test_main_takes_priority_over_agent(self, tmp_path: Path) -> None:
        f = tmp_path / "my_agent.py"
        f.write_text(
            textwrap.dedent("""\
            class FakeAgent:
                def __init__(self, name):
                    self.name = name
                def initiate_chat(self, recipient, message=""):
                    pass

            agent = FakeAgent("researcher")

            def main():
                return "from main"
            """)
        )
        d = discover(f)
        assert d.kind == "main"


class TestLoadYamlConfig:
    def test_loads_valid_yaml(self, yaml_config_file: Path) -> None:
        config = load_yaml_config(yaml_config_file)
        assert config["llm"]["model"] == "gpt-4o"
        assert len(config["agents"]) == 2
        assert config["agents"][0]["name"] == "researcher"

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_yaml_config(tmp_path / "missing.yaml")

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="must be a mapping"):
            load_yaml_config(f)
