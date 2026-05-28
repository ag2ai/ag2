"""Shared fixtures for AG2 CLI tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with basic structure."""
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "__init__.py").touch()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "__init__.py").touch()
    (tmp_path / "teams").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").touch()
    return tmp_path


@pytest.fixture
def agent_file_with_main(tmp_path: Path) -> Path:
    """Create a Python file with a main() function."""
    f = tmp_path / "my_agent.py"
    f.write_text(
        textwrap.dedent("""\
        def main(message="hello"):
            return f"Response to: {message}"
        """)
    )
    return f


@pytest.fixture
def agent_file_with_variable(tmp_path: Path) -> Path:
    """Create a Python file with an 'agent' variable (duck-typed)."""
    f = tmp_path / "my_agent.py"
    f.write_text(
        textwrap.dedent("""\
        class FakeAgent:
            def __init__(self, name):
                self.name = name

            def initiate_chat(self, recipient, message=""):
                return f"{self.name} says: {message}"

        agent = FakeAgent("researcher")
        """)
    )
    return f


@pytest.fixture
def agent_file_with_agents_list(tmp_path: Path) -> Path:
    """Create a Python file with an 'agents' list."""
    f = tmp_path / "my_team.py"
    f.write_text(
        textwrap.dedent("""\
        class FakeAgent:
            def __init__(self, name):
                self.name = name

            def initiate_chat(self, recipient, message=""):
                return f"{self.name} says: {message}"

        agents = [FakeAgent("alice"), FakeAgent("bob")]
        """)
    )
    return f


@pytest.fixture
def agent_file_empty(tmp_path: Path) -> Path:
    """Create a Python file with no discoverable agents."""
    f = tmp_path / "empty.py"
    f.write_text("x = 42\n")
    return f


@pytest.fixture
def yaml_config_file(tmp_path: Path) -> Path:
    """Create a YAML agent config file."""
    f = tmp_path / "team.yaml"
    f.write_text(
        textwrap.dedent("""\
        llm:
          model: gpt-4o

        agents:
          - name: researcher
            system_message: "You research topics."
          - name: writer
            system_message: "You write reports."

        team:
          pattern: auto
          max_rounds: 10
        """)
    )
    return f


@pytest.fixture
def eval_yaml_file(tmp_path: Path) -> Path:
    """Create a YAML eval cases file."""
    f = tmp_path / "cases.yaml"
    f.write_text(
        textwrap.dedent("""\
        name: "test-suite"
        description: "Test evaluation suite"

        cases:
          - name: "basic_test"
            input: "What is the capital of France?"
            assertions:
              - type: contains
                value: "Paris"
              - type: max_turns
                value: 5

          - name: "length_test"
            input: "Write a short paragraph."
            assertions:
              - type: min_length
                value: 10
              - type: max_length
                value: 5000
              - type: no_error
        """)
    )
    return f
