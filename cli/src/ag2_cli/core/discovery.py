"""Agent discovery — find runnable agents in Python files and YAML configs.

Discovery order for Python files:
  1. main() function → call directly
  2. 'agent' or 'team' variable → initiate_chat()
  3. 'agents' list → wrap in GroupChat
  4. Any ConversableAgent subclass instance → use it
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass
class DiscoveredAgent:
    """Result of agent discovery in a file."""

    kind: str  # "main", "agent", "agents"
    source_file: Path
    main_fn: Callable[..., Any] | None = None
    agent: Any = None
    agents: list[Any] = field(default_factory=list)
    agent_names: list[str] = field(default_factory=list)


def import_agent_file(path: Path) -> ModuleType:
    """Import a Python file as a module."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")
    if not path.suffix == ".py":
        raise ValueError(f"Expected a .py file, got: {path}")

    module_name = f"_ag2_user_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")

    module = importlib.util.module_from_spec(spec)
    # Add parent directory to sys.path so relative imports work
    parent = str(path.parent)
    path_added = False
    if parent not in sys.path:
        sys.path.insert(0, parent)
        path_added = True
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        # Remove the broken module so it doesn't pollute sys.modules
        sys.modules.pop(module_name, None)
        raise
    finally:
        if path_added:
            with contextlib.suppress(ValueError):
                sys.path.remove(parent)
    return module


def _is_agent_instance(obj: Any) -> bool:
    """Check if an object is a ConversableAgent instance (or duck-types as one)."""
    try:
        from autogen import ConversableAgent

        if isinstance(obj, ConversableAgent):
            return True
    except ImportError:
        pass
    # Fall back to duck-typing
    return hasattr(obj, "initiate_chat") and hasattr(obj, "name")


def _get_agent_name(obj: Any) -> str:
    """Get the name of an agent-like object."""
    return getattr(obj, "name", str(type(obj).__name__))


def discover(path: Path) -> DiscoveredAgent:
    """Discover runnable agents in a Python file.

    Raises ValueError if no runnable agent is found.
    """
    module = import_agent_file(path)

    # 1. main() function
    main_fn = getattr(module, "main", None)
    if main_fn is not None and callable(main_fn):
        return DiscoveredAgent(kind="main", source_file=path, main_fn=main_fn)

    # 2. 'agent' or 'team' variable
    for var_name in ("agent", "team"):
        obj = getattr(module, var_name, None)
        if obj is not None and _is_agent_instance(obj):
            return DiscoveredAgent(
                kind="agent",
                source_file=path,
                agent=obj,
                agent_names=[_get_agent_name(obj)],
            )

    # 3. 'agents' list
    agents_list = getattr(module, "agents", None)
    if agents_list is not None and isinstance(agents_list, (list, tuple)):
        valid = [a for a in agents_list if _is_agent_instance(a)]
        if valid:
            return DiscoveredAgent(
                kind="agents",
                source_file=path,
                agents=valid,
                agent_names=[_get_agent_name(a) for a in valid],
            )

    # 4. Any ConversableAgent instance in module namespace
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if _is_agent_instance(obj):
            return DiscoveredAgent(
                kind="agent",
                source_file=path,
                agent=obj,
                agent_names=[_get_agent_name(obj)],
            )

    raise ValueError(
        f"No runnable agent found in {path}. Define main(), or a variable named 'agent', 'team', or 'agents'."
    )


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML agent configuration file."""
    import yaml

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping, got {type(config).__name__}")

    return config


def build_agents_from_yaml(config: dict[str, Any]) -> DiscoveredAgent:
    """Build AG2 agents from a YAML configuration.

    Expected format:
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
    """
    try:
        from autogen import AssistantAgent, LLMConfig
    except ImportError as exc:
        raise ImportError("ag2 is required to run YAML configs. Install with: pip install ag2") from exc

    # Build LLM config
    llm_section = config.get("llm", {})
    model = llm_section.get("model", "gpt-4o")
    llm_config = LLMConfig({"model": model, **{k: v for k, v in llm_section.items() if k != "model"}})

    # Build agents
    agent_defs = config.get("agents", [])
    if not agent_defs:
        raise ValueError("YAML config must have an 'agents' list")

    agents = []
    for agent_def in agent_defs:
        name = agent_def.get("name", f"agent_{len(agents)}")
        system_message = agent_def.get("system_message", f"You are {name}.")
        agent = AssistantAgent(name=name, system_message=system_message, llm_config=llm_config)
        agents.append(agent)

    if len(agents) == 1:
        return DiscoveredAgent(
            kind="agent",
            source_file=Path("<yaml>"),
            agent=agents[0],
            agent_names=[agents[0].name],
        )

    return DiscoveredAgent(
        kind="agents",
        source_file=Path("<yaml>"),
        agents=agents,
        agent_names=[a.name for a in agents],
    )
