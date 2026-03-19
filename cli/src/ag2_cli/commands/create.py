"""ag2 create — scaffold projects, agents, tools, and teams."""

from __future__ import annotations

from pathlib import Path

import typer

from ..ui import console

app = typer.Typer(
    help="Scaffold new AG2 projects, agents, tools, and teams.",
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_PYPROJECT_TEMPLATE = """\
[project]
name = "{name}"
version = "0.1.0"
description = "AG2 multi-agent application"
requires-python = ">=3.10"
dependencies = [
    "ag2>=0.11",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
"""

_ENV_EXAMPLE = """\
# LLM provider API keys — uncomment and fill in the ones you use
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
# GOOGLE_API_KEY=...
"""

_GITIGNORE = """\
__pycache__/
*.py[cod]
.env
.venv/
dist/
*.egg-info/
.pytest_cache/
"""

_MAIN_PY_TEMPLATE = """\
\"\"\"Entry point for ag2 run / ag2 chat.\"\"\"

import asyncio

from agents.assistant import assistant


async def main(message: str = "Hello! What can you help me with?"):
    from autogen import UserProxyAgent

    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    result = user.initiate_chat(assistant, message=message)
    return result


if __name__ == "__main__":
    asyncio.run(main())
"""

_AGENT_TEMPLATE = """\
\"\"\"Agent: {name}\"\"\"

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
    {var_name} = AssistantAgent(
        name="{name}",
        system_message={system_message},
    )
"""

_AGENT_WITH_TOOLS_TEMPLATE = """\
\"\"\"Agent: {name}\"\"\"

from autogen import AssistantAgent, LLMConfig

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
    {var_name} = AssistantAgent(
        name="{name}",
        system_message={system_message},
    )

# Tool registration
{tool_imports}
"""

_TOOL_TEMPLATE = """\
\"\"\"Tool: {name}\"\"\"

from autogen.tools import tool


@tool(name="{func_name}", description="{description}")
def {func_name}({params}) -> str:
    \"\"\"{docstring}

    Args:
{args_doc}

    Returns:
        Result as a formatted string.
    \"\"\"
    # TODO: Implement this tool
    raise NotImplementedError("Implement {func_name}")
"""

_TEAM_TEMPLATE = """\
\"\"\"Team: {name}\"\"\"

from autogen import AssistantAgent, LLMConfig
from autogen.agentchat.group import run_group_chat
from autogen.agentchat.group.patterns.pattern import {pattern_class}

config = LLMConfig(api_type="openai", model="gpt-4o")

with config:
{agent_definitions}

agents = [{agent_list}]


async def main(message: str = "Hello team!"):
    pattern = {pattern_class}(
        initial_agent={first_agent},
        agents=agents,
    )
    result = await run_group_chat(
        pattern=pattern,
        messages=message,
        max_rounds=10,
    )
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
"""

_TEST_TEMPLATE = """\
\"\"\"Basic agent tests.\"\"\"

import pytest


def test_agent_importable():
    \"\"\"Verify agent module can be imported.\"\"\"
    from agents.assistant import assistant

    assert assistant is not None
    assert assistant.name == "assistant"
"""

_TEMPLATES = {
    "blank": {
        "description": "Minimal starter project",
        "agents": [("assistant", "You are a helpful assistant.")],
    },
    "research-team": {
        "description": "Web research + report writing team",
        "agents": [
            ("researcher", "You research topics thoroughly using available tools."),
            ("writer", "You write clear, concise reports from research provided to you."),
        ],
    },
    "rag-chatbot": {
        "description": "RAG-enabled chatbot",
        "agents": [("assistant", "You are a helpful assistant that answers questions using retrieved context.")],
    },
}

PATTERN_MAP = {
    "auto": "AutoPattern",
    "round-robin": "RoundRobinPattern",
    "round_robin": "RoundRobinPattern",
    "random": "RandomPattern",
    "manual": "ManualPattern",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_var_name(name: str) -> str:
    """Convert a human name to a valid Python variable name."""
    return name.replace("-", "_").replace(" ", "_").lower()


def _write_file(path: Path, content: str) -> None:
    """Write a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("project")
def create_project(
    name: str = typer.Argument(..., help="Project name."),
    template: str = typer.Option("blank", "--template", "-t", help="Project template to use."),
) -> None:
    """Scaffold a new AG2 project with best-practice structure.

    [dim]Examples:[/dim]
      [command]ag2 create project my-research-bot[/command]
      [command]ag2 create project my-app --template research-team[/command]

    [dim]Templates: blank, research-team, rag-chatbot[/dim]
    """
    tpl = _TEMPLATES.get(template)
    if tpl is None:
        console.print(f"[error]Unknown template: {template}[/error]")
        console.print("Available templates:")
        for tname, tinfo in _TEMPLATES.items():
            console.print(f"  [command]{tname}[/command] — {tinfo['description']}")
        raise typer.Exit(1)

    project_dir = Path.cwd() / name
    if project_dir.exists():
        console.print(f"[error]Directory already exists: {name}[/error]")
        raise typer.Exit(1)

    console.print(f"\n[heading]Creating project:[/heading] {name} (template: {template})\n")

    # Core files
    _write_file(project_dir / "pyproject.toml", _PYPROJECT_TEMPLATE.format(name=name))
    _write_file(project_dir / ".env.example", _ENV_EXAMPLE)
    _write_file(project_dir / ".gitignore", _GITIGNORE)

    # Agents
    _write_file(project_dir / "agents" / "__init__.py", "")
    for agent_name, system_msg in tpl["agents"]:
        var = _to_var_name(agent_name)
        _write_file(
            project_dir / "agents" / f"{var}.py",
            _AGENT_TEMPLATE.format(
                name=agent_name,
                var_name=var,
                system_message=repr(system_msg),
            ),
        )

    # Tools
    _write_file(project_dir / "tools" / "__init__.py", "")

    # Tests
    _write_file(project_dir / "tests" / "__init__.py", "")
    _write_file(project_dir / "tests" / "test_agents.py", _TEST_TEMPLATE)

    # Main entry point
    _write_file(project_dir / "main.py", _MAIN_PY_TEMPLATE)

    # Count generated files
    files = list(project_dir.rglob("*"))
    file_count = sum(1 for f in files if f.is_file())

    console.print(f"  [success]✓[/success] Created [path]{project_dir}[/path] ({file_count} files)")
    console.print()
    console.print("  Next steps:")
    console.print(f"    [command]cd {name}[/command]")
    console.print("    [command]pip install -e .[/command]")
    console.print("    [command]ag2 run main.py --message \"Hello!\"[/command]")
    console.print()


@app.command("agent")
def create_agent(
    name: str = typer.Argument(..., help="Agent name."),
    tools: str | None = typer.Option(None, "--tools", help="Comma-separated tool names to include."),
    from_description: str | None = typer.Option(
        None, "--from-description", help="Generate agent from natural language description (AI-powered)."
    ),
) -> None:
    """Scaffold a new agent with boilerplate and tool wiring.

    [dim]Examples:[/dim]
      [command]ag2 create agent researcher --tools web-search,arxiv[/command]
      [command]ag2 create agent writer[/command]
    """
    if from_description:
        console.print("[warning]--from-description (AI generation) is coming soon.[/warning]")
        raise typer.Exit(0)

    var = _to_var_name(name)
    system_msg = f"You are {name}. You help users by completing tasks thoroughly and accurately."

    # Determine output path
    agents_dir = Path.cwd() / "agents"
    if agents_dir.is_dir():
        out_path = agents_dir / f"{var}.py"
    else:
        out_path = Path.cwd() / f"{var}.py"

    if out_path.exists():
        console.print(f"[error]File already exists: {out_path}[/error]")
        raise typer.Exit(1)

    if tools:
        tool_names = [t.strip() for t in tools.split(",")]
        tool_imports = "\n".join(
            f"# from tools.{_to_var_name(t)} import {_to_var_name(t)}_tool  # TODO: create tool"
            for t in tool_names
        )
        tool_imports += "\n# Register tools:\n"
        tool_imports += "\n".join(
            f"# {_to_var_name(t)}_tool.register_tool({var})" for t in tool_names
        )
        content = _AGENT_WITH_TOOLS_TEMPLATE.format(
            name=name,
            var_name=var,
            system_message=repr(system_msg),
            tool_imports=tool_imports,
        )
    else:
        content = _AGENT_TEMPLATE.format(
            name=name,
            var_name=var,
            system_message=repr(system_msg),
        )

    _write_file(out_path, content)
    console.print(f"  [success]✓[/success] Created [path]{out_path}[/path]")


@app.command("tool")
def create_tool(
    name: str = typer.Argument(..., help="Tool name."),
    description: str | None = typer.Option(None, "--description", "-d", help="Tool description."),
) -> None:
    """Scaffold a new AG2 tool with proper typing and registration.

    [dim]Examples:[/dim]
      [command]ag2 create tool stock-price --description "Fetch real-time stock prices"[/command]
    """
    func_name = _to_var_name(name)
    desc = description or f"Tool: {name}"

    # Determine output path
    tools_dir = Path.cwd() / "tools"
    if tools_dir.is_dir():
        out_path = tools_dir / f"{func_name}.py"
    else:
        out_path = Path.cwd() / f"{func_name}.py"

    if out_path.exists():
        console.print(f"[error]File already exists: {out_path}[/error]")
        raise typer.Exit(1)

    content = _TOOL_TEMPLATE.format(
        name=name,
        func_name=func_name,
        description=desc,
        params="query: str",
        docstring=desc,
        args_doc="        query: Input query or parameter.",
    )

    _write_file(out_path, content)
    console.print(f"  [success]✓[/success] Created [path]{out_path}[/path]")


@app.command("team")
def create_team(
    name: str = typer.Argument(..., help="Team name."),
    pattern: str = typer.Option("auto", "--pattern", "-p", help="Orchestration pattern (auto, round-robin, random)."),
    agents: str | None = typer.Option(None, "--agents", "-a", help="Comma-separated agent names."),
) -> None:
    """Scaffold a multi-agent team with orchestration pattern.

    [dim]Examples:[/dim]
      [command]ag2 create team code-review --pattern round-robin --agents reviewer,tester,merger[/command]
    """
    pattern_class = PATTERN_MAP.get(pattern)
    if pattern_class is None:
        console.print(f"[error]Unknown pattern: {pattern}[/error]")
        console.print(f"Available: {', '.join(PATTERN_MAP.keys())}")
        raise typer.Exit(1)

    agent_names = [a.strip() for a in agents.split(",")] if agents else ["agent_a", "agent_b"]

    # Determine output path
    teams_dir = Path.cwd() / "teams"
    out_dir = teams_dir if teams_dir.is_dir() else Path.cwd()
    var = _to_var_name(name)
    out_path = out_dir / f"{var}.py"

    if out_path.exists():
        console.print(f"[error]File already exists: {out_path}[/error]")
        raise typer.Exit(1)

    # Build agent definitions
    agent_defs = []
    for aname in agent_names:
        avar = _to_var_name(aname)
        agent_defs.append(
            f'    {avar} = AssistantAgent(\n'
            f'        name="{aname}",\n'
            f'        system_message="You are {aname}.",\n'
            f'    )'
        )

    agent_list = ", ".join(_to_var_name(a) for a in agent_names)
    first_agent = _to_var_name(agent_names[0])

    content = _TEAM_TEMPLATE.format(
        name=name,
        pattern_class=pattern_class,
        agent_definitions="\n".join(agent_defs),
        agent_list=agent_list,
        first_agent=first_agent,
    )

    _write_file(out_path, content)
    console.print(f"  [success]✓[/success] Created [path]{out_path}[/path]")
