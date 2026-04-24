"""ag2 create — scaffold projects, agents, tools, teams, and artifacts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import typer

from ..ui import console

app = typer.Typer(
    help="Scaffold new AG2 projects, agents, tools, teams, and artifacts.",
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

config = LLMConfig({{"model": "gpt-4o"}})

{var_name} = AssistantAgent(
    name="{name}",
    system_message={system_message},
    llm_config=config,
)
"""

_AGENT_WITH_TOOLS_TEMPLATE = """\
\"\"\"Agent: {name}\"\"\"

from autogen import AssistantAgent, LLMConfig

config = LLMConfig({{"model": "gpt-4o"}})

{var_name} = AssistantAgent(
    name="{name}",
    system_message={system_message},
    llm_config=config,
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
from autogen.agentchat import run_group_chat
from autogen.agentchat.group.patterns.pattern import {pattern_class}

config = LLMConfig({{"model": "gpt-4o"}})

{agent_definitions}

agents = [{agent_list}]


def main(message: str = "Hello team!"):
    pattern = {pattern_class}(
        initial_agent={first_agent},
        agents=agents,
    )
    response = run_group_chat(
        pattern=pattern,
        messages=message,
        max_rounds=10,
    )
    response.process()
    return response.summary
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
    "fullstack-agentic": {
        "description": "Full-stack app with agent backend",
        "agents": [
            ("planner", "You plan features and break them into implementable tasks."),
            ("coder", "You write clean, production-ready Python code."),
            ("reviewer", "You review code for bugs, security issues, and best practices."),
        ],
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
# LLM-powered generation helpers
# ---------------------------------------------------------------------------

_LLM_YAML = """\
# LLM configuration — ag2 run / ag2 chat will pick this up via your env vars.
# Set the API key for your provider in .env or your shell environment.
model: gpt-4o
"""

_WEB_SEARCH_TOOL = """\
\"\"\"Example tool: web search.\"\"\"

from autogen.tools import tool


@tool(name="web_search", description="Search the web for information")
def web_search(query: str) -> str:
    \"\"\"Search the web and return results.

    Args:
        query: Search query string.

    Returns:
        Search results as a formatted string.
    \"\"\"
    # TODO: Implement with your preferred search API (e.g., Tavily, SerpAPI, Brave)
    raise NotImplementedError("Connect to your preferred search API")
"""


def _detect_generation_model() -> str | None:
    """Auto-detect an available LLM model from environment API keys."""
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4o"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude-sonnet-4-20250514"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini-2.0-flash"
    return None


def _llm_generate(prompt: str, system_message: str) -> str:
    """Make a single LLM call and return the response text.

    Uses AG2's own agents to call an LLM — requires ag2 and an API key.
    """
    try:
        import autogen
    except ImportError:
        console.print("[error]ag2 is not installed.[/error]")
        console.print("Install with: [command]pip install ag2[/command]")
        raise typer.Exit(1)

    model = _detect_generation_model()
    if not model:
        console.print("[error]No LLM API key found in environment.[/error]")
        console.print("Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
        raise typer.Exit(1)

    console.print(f"[dim]Using {model} for generation...[/dim]")

    llm_config = autogen.LLMConfig({"model": model})
    generator = autogen.AssistantAgent(
        name="generator",
        system_message=system_message,
        llm_config=llm_config,
    )
    user = autogen.UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
    )
    result = user.initiate_chat(generator, message=prompt, silent=True)

    # Extract response from chat history
    for msg in reversed(result.chat_history):
        if msg.get("name") == "generator" or msg.get("role") == "assistant":
            return msg.get("content", "")
    return result.summary or ""


def _parse_json_response(text: str) -> dict:
    """Extract and parse JSON from an LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to extract from markdown code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try to find a JSON object in the text
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    console.print("[error]Could not parse LLM response as JSON.[/error]")
    console.print("[dim]Raw response:[/dim]")
    console.print(text[:500])
    raise typer.Exit(1)


_AGENT_GEN_SYSTEM = """\
You are an expert AG2 (AutoGen) developer. You generate agent specifications as JSON.

AG2 uses this pattern for agents:
```python
from autogen import AssistantAgent, LLMConfig
config = LLMConfig({{"model": "gpt-4o"}})
agent = AssistantAgent(name="agent_name", system_message="...", llm_config=config)
```

Tools use:
```python
from autogen.tools import tool
@tool(name="tool_name", description="...")
def tool_name(param: str) -> str:
    ...
```

Always respond with ONLY a valid JSON object. No markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("project")
def create_project(
    name: str | None = typer.Argument(None, help="Project name (optional with --from-description)."),
    template: str = typer.Option("blank", "--template", "-t", help="Project template to use."),
    from_description: str | None = typer.Option(
        None, "--from-description", help="Generate project from natural language description (AI-powered)."
    ),
) -> None:
    """Scaffold a new AG2 project with best-practice structure.

    [dim]Examples:[/dim]
      [command]ag2 create project my-research-bot[/command]
      [command]ag2 create project my-app --template research-team[/command]
      [command]ag2 create project --from-description "A Slack bot that summarizes channels"[/command]

    [dim]Templates: blank, research-team, rag-chatbot, fullstack-agentic[/dim]
    """
    if from_description:
        _create_project_from_description(from_description, name)
        return

    if not name:
        console.print("[error]Project name is required (or use --from-description).[/error]")
        raise typer.Exit(1)

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

    # Config
    _write_file(project_dir / "config" / "llm.yaml", _LLM_YAML)

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
    _write_file(project_dir / "tools" / "web_search.py", _WEB_SEARCH_TOOL)

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
    console.print('    [command]ag2 run main.py --message "Hello!"[/command]')
    console.print()


def _create_project_from_description(description: str, name: str | None) -> None:
    """Generate a full project from a natural language description using an LLM."""
    console.print("\n[heading]Generating project from description...[/heading]\n")

    prompt = (
        f"Generate an AG2 project specification for:\n{description}\n\n"
        "Return a JSON object with:\n"
        "{\n"
        '  "name": "project-name-slug",\n'
        '  "description": "One-line project description",\n'
        '  "agents": [\n'
        '    {"name": "agent_name", "system_message": "Detailed system prompt", "tools": ["tool_name"]}\n'
        "  ],\n"
        '  "tools": [\n'
        '    {"name": "tool_name", "description": "What the tool does",\n'
        '     "params": [{"name": "param_name", "type": "str", "description": "..."}]}\n'
        "  ],\n"
        '  "pattern": "auto"\n'
        "}\n\n"
        "Choose pattern from: auto, round-robin, random.\n"
        "Use auto for most cases. Include 1-4 agents and relevant tools.\n"
        "Write detailed, specific system messages for each agent."
    )

    response = _llm_generate(prompt, _AGENT_GEN_SYSTEM)
    spec = _parse_json_response(response)

    project_name = name or spec.get("name", "my-agent-project")
    project_dir = Path.cwd() / project_name
    if project_dir.exists():
        console.print(f"[error]Directory already exists: {project_name}[/error]")
        raise typer.Exit(1)

    agents = spec.get("agents", [])
    tools = spec.get("tools", [])
    pattern = spec.get("pattern", "auto")

    if not agents:
        console.print("[error]LLM did not generate any agents. Try a more detailed description.[/error]")
        raise typer.Exit(1)

    console.print(f"[dim]Generating project: {project_name}[/dim]")
    console.print(f"[dim]  Agents: {', '.join(a['name'] for a in agents)}[/dim]")
    if tools:
        console.print(f"[dim]  Tools: {', '.join(t['name'] for t in tools)}[/dim]")

    # Core files
    _write_file(project_dir / "pyproject.toml", _PYPROJECT_TEMPLATE.format(name=project_name))
    _write_file(project_dir / ".env.example", _ENV_EXAMPLE)
    _write_file(project_dir / ".gitignore", _GITIGNORE)
    _write_file(project_dir / "config" / "llm.yaml", _LLM_YAML)

    # Generate agents
    _write_file(project_dir / "agents" / "__init__.py", "")
    for agent_def in agents:
        aname = agent_def["name"]
        avar = _to_var_name(aname)
        sys_msg = agent_def.get("system_message", f"You are {aname}.")
        agent_tools = agent_def.get("tools", [])

        if agent_tools:
            tool_imports = "\n".join(f"from tools.{_to_var_name(t)} import {_to_var_name(t)}" for t in agent_tools)
            tool_registers = "\n".join(f"{_to_var_name(t)}.register_tool({avar})" for t in agent_tools)
            content = _AGENT_WITH_TOOLS_TEMPLATE.format(
                name=aname,
                var_name=avar,
                system_message=repr(sys_msg),
                tool_imports=f"{tool_imports}\n\n{tool_registers}",
            )
        else:
            content = _AGENT_TEMPLATE.format(
                name=aname,
                var_name=avar,
                system_message=repr(sys_msg),
            )
        _write_file(project_dir / "agents" / f"{avar}.py", content)

    # Generate tools
    _write_file(project_dir / "tools" / "__init__.py", "")
    for tool_def in tools:
        tname = tool_def["name"]
        tfunc = _to_var_name(tname)
        tdesc = tool_def.get("description", f"Tool: {tname}")
        tparams = tool_def.get("params", [{"name": "query", "type": "str", "description": "Input query."}])

        params_str = ", ".join(f"{p['name']}: {p.get('type', 'str')}" for p in tparams)
        args_doc = "\n".join(f"        {p['name']}: {p.get('description', '')}" for p in tparams)

        content = _TOOL_TEMPLATE.format(
            name=tname,
            func_name=tfunc,
            description=tdesc,
            params=params_str,
            docstring=tdesc,
            args_doc=args_doc,
        )
        _write_file(project_dir / "tools" / f"{tfunc}.py", content)

    # Generate main.py
    if len(agents) == 1:
        avar = _to_var_name(agents[0]["name"])
        main_content = (
            f'"""Entry point for ag2 run / ag2 chat."""\n\n'
            f"import asyncio\n\n"
            f"from agents.{avar} import {avar}\n\n\n"
            f'async def main(message: str = "Hello! What can you help me with?"):\n'
            f"    from autogen import UserProxyAgent\n\n"
            f'    user = UserProxyAgent(\n        name="user",\n'
            f'        human_input_mode="NEVER",\n'
            f"        max_consecutive_auto_reply=0,\n"
            f"        code_execution_config=False,\n    )\n"
            f"    result = user.initiate_chat({avar}, message=message)\n"
            f"    return result\n\n\n"
            f'if __name__ == "__main__":\n    asyncio.run(main())\n'
        )
    else:
        pattern_class = PATTERN_MAP.get(pattern, "AutoPattern")
        imports = "\n".join(f"from agents.{_to_var_name(a['name'])} import {_to_var_name(a['name'])}" for a in agents)
        agent_list = ", ".join(_to_var_name(a["name"]) for a in agents)
        first = _to_var_name(agents[0]["name"])
        main_content = (
            f'"""Entry point for ag2 run / ag2 chat."""\n\n'
            f"import asyncio\n\n"
            f"from autogen.agentchat.group import run_group_chat\n"
            f"from autogen.agentchat.group.patterns.pattern import {pattern_class}\n\n"
            f"{imports}\n\n"
            f"agents = [{agent_list}]\n\n\n"
            f'async def main(message: str = "Hello team!"):\n'
            f"    pattern = {pattern_class}(\n"
            f"        initial_agent={first},\n"
            f"        agents=agents,\n"
            f"    )\n"
            f"    result = await run_group_chat(\n"
            f"        pattern=pattern,\n"
            f"        messages=message,\n"
            f"        max_rounds=10,\n"
            f"    )\n"
            f"    return result\n\n\n"
            f'if __name__ == "__main__":\n    asyncio.run(main())\n'
        )
    _write_file(project_dir / "main.py", main_content)

    # Tests
    _write_file(project_dir / "tests" / "__init__.py", "")
    first_var = _to_var_name(agents[0]["name"])
    first_name = agents[0]["name"]
    test_content = (
        f'"""Basic agent tests."""\n\nimport pytest\n\n\n'
        f"def test_agent_importable():\n"
        f'    """Verify agent module can be imported."""\n'
        f"    from agents.{first_var} import {first_var}\n\n"
        f"    assert {first_var} is not None\n"
        f'    assert {first_var}.name == "{first_name}"\n'
    )
    _write_file(project_dir / "tests" / "test_agents.py", test_content)

    # Summary
    file_count = sum(1 for f in project_dir.rglob("*") if f.is_file())
    console.print(f"\n  [success]✓[/success] Created [path]{project_dir}[/path] ({file_count} files)")
    console.print()
    console.print("  Next steps:")
    console.print(f"    [command]cd {project_name}[/command]")
    console.print("    [command]pip install -e .[/command]")
    console.print('    [command]ag2 run main.py --message "Hello!"[/command]')
    console.print()


@app.command("agent")
def create_agent(
    name: str | None = typer.Argument(None, help="Agent name (optional with --from-description)."),
    tools: str | None = typer.Option(None, "--tools", help="Comma-separated tool names to include."),
    from_description: str | None = typer.Option(
        None, "--from-description", help="Generate agent from natural language description (AI-powered)."
    ),
) -> None:
    """Scaffold a new agent with boilerplate and tool wiring.

    [dim]Examples:[/dim]
      [command]ag2 create agent researcher --tools web-search,arxiv[/command]
      [command]ag2 create agent writer[/command]
      [command]ag2 create agent --from-description "An agent that monitors HN for AI papers"[/command]
    """
    if from_description:
        _create_agent_from_description(from_description, name)
        return

    if not name:
        console.print("[error]Agent name is required (or use --from-description).[/error]")
        raise typer.Exit(1)

    var = _to_var_name(name)
    system_msg = f"You are {name}. You help users by completing tasks thoroughly and accurately."

    # Determine output path
    agents_dir = Path.cwd() / "agents"
    out_path = agents_dir / f"{var}.py" if agents_dir.is_dir() else Path.cwd() / f"{var}.py"

    if out_path.exists():
        console.print(f"[error]File already exists: {out_path}[/error]")
        raise typer.Exit(1)

    if tools:
        tool_names = [t.strip() for t in tools.split(",")]
        tool_imports = "\n".join(
            f"# from tools.{_to_var_name(t)} import {_to_var_name(t)}_tool  # TODO: create tool" for t in tool_names
        )
        tool_imports += "\n# Register tools:\n"
        tool_imports += "\n".join(f"# {_to_var_name(t)}_tool.register_tool({var})" for t in tool_names)
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


def _create_agent_from_description(description: str, name: str | None) -> None:
    """Generate an agent and its tools from a natural language description."""
    console.print("\n[heading]Generating agent from description...[/heading]\n")

    prompt = (
        f"Generate an AG2 agent specification for:\n{description}\n\n"
        "Return a JSON object with:\n"
        "{\n"
        '  "name": "snake_case_name",\n'
        '  "system_message": "Detailed system prompt for the agent",\n'
        '  "tools": [\n'
        '    {"name": "tool_name", "description": "What the tool does",\n'
        '     "params": [{"name": "param_name", "type": "str", "description": "..."}]}\n'
        "  ]\n"
        "}\n\n"
        "Write a detailed, specific system message. Include 0-4 tools as needed."
    )

    response = _llm_generate(prompt, _AGENT_GEN_SYSTEM)
    spec = _parse_json_response(response)

    agent_name = name or spec.get("name", "assistant")
    var = _to_var_name(agent_name)
    system_msg = spec.get("system_message", f"You are {agent_name}.")
    tool_specs = spec.get("tools", [])

    console.print(f"[dim]Agent: {agent_name}[/dim]")
    if tool_specs:
        console.print(f"[dim]Tools: {', '.join(t['name'] for t in tool_specs)}[/dim]")

    # Write agent file
    agents_dir = Path.cwd() / "agents"
    out_path = agents_dir / f"{var}.py" if agents_dir.is_dir() else Path.cwd() / f"{var}.py"

    if out_path.exists():
        console.print(f"[error]File already exists: {out_path}[/error]")
        raise typer.Exit(1)

    if tool_specs:
        tool_imports = "\n".join(
            f"from tools.{_to_var_name(t['name'])} import {_to_var_name(t['name'])}" for t in tool_specs
        )
        tool_registers = "\n".join(f"{_to_var_name(t['name'])}.register_tool({var})" for t in tool_specs)
        content = _AGENT_WITH_TOOLS_TEMPLATE.format(
            name=agent_name,
            var_name=var,
            system_message=repr(system_msg),
            tool_imports=f"{tool_imports}\n\n{tool_registers}",
        )
    else:
        content = _AGENT_TEMPLATE.format(
            name=agent_name,
            var_name=var,
            system_message=repr(system_msg),
        )

    _write_file(out_path, content)
    console.print(f"  [success]✓[/success] Created [path]{out_path}[/path]")

    # Write tool stubs
    tools_dir = Path.cwd() / "tools"
    for tool_def in tool_specs:
        tname = tool_def["name"]
        tfunc = _to_var_name(tname)
        tdesc = tool_def.get("description", f"Tool: {tname}")
        tparams = tool_def.get("params", [{"name": "query", "type": "str", "description": "Input query."}])

        params_str = ", ".join(f"{p['name']}: {p.get('type', 'str')}" for p in tparams)
        args_doc = "\n".join(f"        {p['name']}: {p.get('description', '')}" for p in tparams)

        tool_content = _TOOL_TEMPLATE.format(
            name=tname,
            func_name=tfunc,
            description=tdesc,
            params=params_str,
            docstring=tdesc,
            args_doc=args_doc,
        )
        tool_path = tools_dir / f"{tfunc}.py" if tools_dir.is_dir() else Path.cwd() / f"{tfunc}.py"
        if not tool_path.exists():
            _write_file(tool_path, tool_content)
            console.print(f"  [success]✓[/success] Created [path]{tool_path}[/path]")


@app.command("tool")
def create_tool(
    name: str | None = typer.Argument(None, help="Tool name (optional with --from-openapi / --from-module)."),
    description: str | None = typer.Option(None, "--description", "-d", help="Tool description."),
    from_openapi: str | None = typer.Option(
        None, "--from-openapi", help="Generate tools from an OpenAPI spec (URL or file path)."
    ),
    from_module: str | None = typer.Option(
        None, "--from-module", help="Generate tools from a Python module (e.g., 'pandas')."
    ),
    functions: str | None = typer.Option(
        None, "--functions", "-f", help="Comma-separated function names (with --from-module)."
    ),
) -> None:
    """Scaffold a new AG2 tool with proper typing and registration.

    [dim]Examples:[/dim]
      [command]ag2 create tool stock-price --description "Fetch real-time stock prices"[/command]
      [command]ag2 create tool --from-openapi https://api.example.com/openapi.json[/command]
      [command]ag2 create tool --from-module pandas --functions read_csv,describe[/command]
    """
    if from_openapi:
        _create_tool_from_openapi(from_openapi, name)
        return

    if from_module:
        func_list = [f.strip() for f in functions.split(",") if f.strip()] if functions else None
        _create_tool_from_module(from_module, func_list, name)
        return

    if not name:
        console.print("[error]Tool name is required (or use --from-openapi / --from-module).[/error]")
        raise typer.Exit(1)

    func_name = _to_var_name(name)
    desc = description or f"Tool: {name}"

    # Determine output path
    tools_dir = Path.cwd() / "tools"
    out_path = tools_dir / f"{func_name}.py" if tools_dir.is_dir() else Path.cwd() / f"{func_name}.py"

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


def _create_tool_from_openapi(source: str, name: str | None) -> None:
    """Generate tool files from an OpenAPI spec, reusing proxy module logic."""
    from .proxy import _generate_tool_file, _load_openapi_spec, _parse_openapi_spec

    console.print(f"[dim]Loading OpenAPI spec from {source}...[/dim]")
    try:
        spec = _load_openapi_spec(source)
    except Exception as exc:
        console.print(f"[error]Failed to load spec: {exc}[/error]")
        raise typer.Exit(1)

    tools = _parse_openapi_spec(spec)
    if not tools:
        console.print("[error]No endpoints found in the spec.[/error]")
        raise typer.Exit(1)

    output_name = _to_var_name(name) if name else "api_tools"
    tools_dir = Path.cwd() / "tools"
    out_path = tools_dir / f"{output_name}.py" if tools_dir.is_dir() else Path.cwd() / f"{output_name}.py"

    _generate_tool_file(tools, out_path)
    console.print(f"  [success]✓[/success] Generated {len(tools)} tools in [path]{out_path}[/path]")


def _create_tool_from_module(module_name: str, func_names: list[str] | None, name: str | None) -> None:
    """Generate tool files from a Python module, reusing proxy module logic."""
    from .proxy import _generate_tool_file, _inspect_module_functions

    console.print(f"[dim]Inspecting module {module_name}...[/dim]")
    tools = _inspect_module_functions(module_name, func_names)
    if not tools:
        console.print(f"[error]No public functions found in {module_name}.[/error]")
        raise typer.Exit(1)

    output_name = _to_var_name(name) if name else f"{_to_var_name(module_name)}_tools"
    tools_dir = Path.cwd() / "tools"
    out_path = tools_dir / f"{output_name}.py" if tools_dir.is_dir() else Path.cwd() / f"{output_name}.py"

    _generate_tool_file(tools, out_path)
    console.print(f"  [success]✓[/success] Generated {len(tools)} tools in [path]{out_path}[/path]")


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
            f'{avar} = AssistantAgent(\n    name="{aname}",\n    system_message="You are {aname}.",\n    llm_config=config,\n)'
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


# ---------------------------------------------------------------------------
# ag2 create artifact
# ---------------------------------------------------------------------------

ARTIFACT_TYPES = ["template", "tool", "dataset", "agent", "skills", "bundle"]


def _artifact_json(name: str, artifact_type: str, **extra: object) -> str:
    """Generate a starter artifact.json."""
    data: dict = {
        "name": name,
        "type": artifact_type,
        "display_name": name.replace("-", " ").title(),
        "description": "",
        "version": "0.1.0",
        "authors": [],
        "license": "Apache-2.0",
        "tags": [],
    }
    data.update(extra)
    return json.dumps(data, indent=2) + "\n"


def _skill_md(name: str, description: str) -> str:
    """Generate a placeholder SKILL.md."""
    return f"""\
---
name: {name}
description: {description}
license: Apache-2.0
---

# {name.replace("-", " ").title()}

TODO: Write skill content here.
"""


def _scaffold_template(name: str, out: Path) -> None:
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "template",
            template={
                "scaffold": "scaffold/",
                "variables": {
                    "project_name": {"prompt": "Project name", "default": f"my-{name}", "transform": "slug"},
                    "description": {"prompt": "Project description", "default": ""},
                },
                "ignore": ["__pycache__", "*.pyc", ".git"],
                "post_install": [],
            },
            skills={"dir": "skills/", "auto_install": True},
        ),
    )
    _write_file(out / "scaffold" / "README.md.tmpl", "# {{ project_name }}\n\n{{ description }}\n")
    _write_file(
        out / "skills" / "rules" / f"{name}-architecture" / "SKILL.md",
        _skill_md(f"{name}-architecture", f"Architecture overview and conventions for {name}"),
    )
    _write_file(
        out / "skills" / "skills" / "add-feature" / "SKILL.md",
        _skill_md("add-feature", f"Step-by-step guide to add features to {name}"),
    )


def _scaffold_tool(name: str, out: Path) -> None:
    slug = _to_var_name(name)
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "tool",
            tool={
                "kind": "ag2",
                "source": "src/",
                "functions": [{"name": slug, "description": ""}],
                "requires": [],
                "install_to": "tools/",
            },
            skills={"dir": "skills/", "auto_install": True},
        ),
    )
    _write_file(out / "src" / "__init__.py", "")
    _write_file(
        out / "src" / f"{slug}.py",
        f'"""Tool: {name}"""\n\n\ndef {slug}(query: str) -> str:\n    raise NotImplementedError\n',
    )
    _write_file(
        out / "tests" / f"test_{slug}.py",
        f'"""Tests for {name}."""\n\nfrom src.{slug} import {slug}\n\n\ndef test_{slug}_placeholder():\n    pass\n',
    )
    _write_file(
        out / "skills" / "skills" / f"integrate-{name}" / "SKILL.md",
        _skill_md(f"integrate-{name}", f"How to register and use the {name} tool with AG2 agents"),
    )


def _scaffold_dataset(name: str, out: Path) -> None:
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "dataset",
            dataset={
                "inline": "data/",
                "remote": [],
                "format": "jsonl",
                "schema": {"fields": []},
                "splits": {"sample": "data/sample.jsonl"},
                "eval_compatible": False,
            },
            skills={"dir": "skills/", "auto_install": True},
        ),
    )
    _write_file(out / "data" / "sample.jsonl", '{"input": "example", "expected": "result"}\n')
    _write_file(
        out / "skills" / "rules" / f"{name}-schema" / "SKILL.md",
        _skill_md(f"{name}-schema", f"Schema and usage guide for the {name} dataset"),
    )


def _scaffold_agent(name: str, out: Path) -> None:
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "agent",
            agent={
                "source": "agent.md",
                "model": "sonnet",
                "tools": [],
                "max_turns": 50,
                "memory": "project",
                "mcp_servers": {},
                "preload_skills": [],
            },
            skills={"dir": "skills/", "auto_install": True},
        ),
    )
    _write_file(out / "agent.md", f"# {name.replace('-', ' ').title()}\n\nYou are {name}. Describe your role here.\n")
    _write_file(
        out / "skills" / "skills" / f"use-{name}" / "SKILL.md",
        _skill_md(f"use-{name}", f"How to use the {name} agent effectively"),
    )


def _scaffold_skills(name: str, out: Path) -> None:
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "skills",
            skills={"dir": ".", "auto_install": True},
        ),
    )
    _write_file(
        out / "rules" / name / "SKILL.md",
        _skill_md(name, f"Conventions and patterns for {name}"),
    )
    _write_file(
        out / "skills" / f"{name}-guide" / "SKILL.md",
        _skill_md(f"{name}-guide", f"Step-by-step guide for {name}"),
    )


def _scaffold_bundle(name: str, out: Path) -> None:
    _write_file(
        out / "artifact.json",
        _artifact_json(
            name,
            "bundle",
            bundle={
                "artifacts": [],
                "install_order": ["skills", "tools", "templates", "datasets", "agents"],
            },
        ),
    )


_SCAFFOLD_FNS = {
    "template": _scaffold_template,
    "tool": _scaffold_tool,
    "dataset": _scaffold_dataset,
    "agent": _scaffold_agent,
    "skills": _scaffold_skills,
    "bundle": _scaffold_bundle,
}


@app.command("artifact")
def create_artifact(
    artifact_type: str = typer.Argument(..., help=f"Artifact type ({', '.join(ARTIFACT_TYPES)})."),
    name: str = typer.Argument(..., help="Artifact name (e.g. my-template)."),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Parent directory for output (default: cwd)."),
) -> None:
    """Scaffold a new artifact for the AG2 artifacts registry.

    [dim]Creates the directory structure, artifact.json, and placeholder skills
    ready for authoring. Publish with [command]ag2 publish artifact[/command].[/dim]

    [dim]Examples:[/dim]
      [command]ag2 create artifact template my-template[/command]
      [command]ag2 create artifact tool web-scraper[/command]
      [command]ag2 create artifact dataset eval-bench[/command]
    """
    if artifact_type not in ARTIFACT_TYPES:
        console.print(f"[error]Unknown artifact type: {artifact_type}[/error]")
        console.print(f"Available types: {', '.join(ARTIFACT_TYPES)}")
        raise typer.Exit(1)

    parent = output_dir or Path.cwd()
    out = parent / name
    if out.exists():
        console.print(f"[error]Directory already exists: {out}[/error]")
        raise typer.Exit(1)

    console.print(f"\n[heading]Creating {artifact_type} artifact:[/heading] {name}\n")

    scaffold_fn = _SCAFFOLD_FNS[artifact_type]
    scaffold_fn(name, out)

    file_count = sum(1 for f in out.rglob("*") if f.is_file())
    console.print(f"  [success]✓[/success] Created [path]{out}[/path] ({file_count} files)")
    console.print()
    console.print("  Next steps:")
    console.print(f"    1. Edit [path]{out / 'artifact.json'}[/path] — fill in description, authors, tags")
    if artifact_type != "bundle":
        console.print(f"    2. Write your skills in [path]{out / 'skills'}[/path]")
    console.print(f"    3. Publish with [command]ag2 publish artifact {out}[/command]")
    console.print()
