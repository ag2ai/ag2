# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Agent manifest loading for team orchestration.

Loads agent definitions from a JSON manifest file and creates worker agents
from three categories:

1. **Remote A2A** (`a2a` key, or legacy `agents`):
   External A2A services served remotely. Each gets a wrapper ConversableAgent
   with an `ask_specialist` tool that delegates work to the remote endpoint.

2. **Local agents** (`local_agents` key):
   Inline-defined agents with system_message + model. Created as regular
   ConversableAgents that do work directly (no delegation wrapper).

3. **Reference agents** (`reference_agents` key):
   AG2 built-in agent classes (e.g. SlackAgent, WebSurferAgent). Imported by
   class name and instantiated with provided args. Supports ${VAR} env var
   expansion in args for credentials.

Manifest format example:
  {
    "a2a": [
      {"id": "...", "name": "RemoteAgent", "url": "https://...", ...}
    ],
    "local_agents": [
      {"name": "writer", "description": "...", "system_message": "...",
       "llm_config": {"model": "claude-haiku-4-5", "api_type": "anthropic"}}
    ],
    "reference_agents": [
      {"name": "slack-bot", "agent_class": "SlackAgent", "args": {"bot_token": "${SLACK_BOT_TOKEN}"}}
    ]
  }
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autogen.agentchat.conversable_agent import ConversableAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent class registry — maps class names to their import modules
# ---------------------------------------------------------------------------

AGENT_CLASSES: dict[str, str] = {
    "SlackAgent": "autogen.agents.experimental",
    "DiscordAgent": "autogen.agents.experimental",
    "TelegramAgent": "autogen.agents.experimental",
    "WebSurferAgent": "autogen.agents.experimental",
    "WikipediaAgent": "autogen.agents.experimental",
    "ReasoningAgent": "autogen.agents.experimental",
    "DeepResearchAgent": "autogen.agents.experimental",
}


# ---------------------------------------------------------------------------
# Manifest models
# ---------------------------------------------------------------------------


class ManifestSkill(BaseModel):
    """A skill advertised by a manifest agent."""

    id: str
    name: str
    description: str = ""


class ManifestAgentContent(BaseModel):
    """Content block within a manifest agent entry."""

    skills: list[ManifestSkill] = Field(default_factory=list)
    organization: str = ""
    version: str = ""


class ManifestAgent(BaseModel):
    """A single remote A2A agent entry from a manifest file."""

    id: str
    name: str
    description: str = ""
    url: str = ""
    platform: str = ""
    framework_type: str = ""
    is_global: bool = False
    content: ManifestAgentContent = Field(default_factory=ManifestAgentContent)
    required_connector_types: list[str] = Field(default_factory=list)


class LocalAgentDef(BaseModel):
    """An inline-defined local agent with system_message and LLM config overrides."""

    name: str
    description: str = ""
    system_message: str = ""
    llm_config: dict[str, Any] = Field(default_factory=dict)


class ReferenceAgentDef(BaseModel):
    """A reference to an AG2 built-in agent class."""

    name: str
    agent_class: str
    description: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    llm_config: dict[str, Any] = Field(default_factory=dict)


class AgentManifest(BaseModel):
    """Top-level manifest supporting three agent categories.

    - ``a2a``: Remote A2A agents (wrapper + ask_specialist delegation)
    - ``local_agents``: Inline-defined ConversableAgents with system_message
    - ``reference_agents``: AG2 built-in classes instantiated by name

    The legacy ``agents`` key is kept for backward compatibility and is merged
    with ``a2a`` during processing.
    """

    a2a: list[ManifestAgent] = Field(default_factory=list)
    agents: list[ManifestAgent] = Field(default_factory=list)  # deprecated alias for a2a
    local_agents: list[LocalAgentDef] = Field(default_factory=list)
    reference_agents: list[ReferenceAgentDef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path) -> AgentManifest:
    """Load an agent manifest from a JSON file.

    Args:
        path: Path to the manifest JSON file.

    Returns:
        Parsed AgentManifest.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the JSON is invalid or doesn't match the schema.
    """
    path = Path(path)
    data = json.loads(path.read_text())
    return AgentManifest.model_validate(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR_NAME}`` references in strings from env vars.

    Works on strings, dicts, and lists. Non-string leaves pass through unchanged.
    Raises ``KeyError`` if an env var is not set.
    """
    if isinstance(value, str):

        def _replace(m: re.Match) -> str:
            var_name = m.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise KeyError(f"Environment variable '{var_name}' is not set")
            return env_val

        return _ENV_VAR_RE.sub(_replace, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _import_agent_class(class_name: str) -> type:
    """Import an AG2 agent class by name from the registry.

    Args:
        class_name: Name of the agent class (e.g. ``"SlackAgent"``).

    Returns:
        The agent class.

    Raises:
        ValueError: If the class name is not in the registry.
        ImportError: If the module cannot be imported.
    """
    module_path = AGENT_CLASSES.get(class_name)
    if module_path is None:
        available = ", ".join(sorted(AGENT_CLASSES.keys()))
        raise ValueError(f"Unknown agent class '{class_name}'. Available classes: {available}")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")
    return cls


def _merge_llm_config(
    base_config: Any,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Merge overrides into a base LLM config, returning a new LLMConfig.

    If no overrides are specified (None or empty dict), returns the base config
    unchanged.  Otherwise, clones the base config and applies overrides.

    Works with both ``LLMConfig`` objects and plain dicts.
    """
    if not overrides:
        return base_config

    from autogen import LLMConfig

    # Extract a flat config dict from the base
    if isinstance(base_config, LLMConfig):
        dumped = base_config.model_dump()
        # LLMConfig.model_dump() returns {"config_list": [{...}]}
        # Extract the first entry as a flat dict for reconstruction
        config_dict = dict(dumped.get("config_list", [{}])[0])
    elif isinstance(base_config, dict):
        config_dict = dict(base_config)
    else:
        config_dict = {}

    config_dict.update(overrides)

    return LLMConfig(config_dict)


# ---------------------------------------------------------------------------
# Build worker agents from manifest
# ---------------------------------------------------------------------------


def _make_agent_description(agent: ManifestAgent) -> str:
    """Build a concise description for the leader to understand an agent's capabilities."""
    desc = agent.description.strip()
    # Truncate long descriptions but keep enough for the leader to match tasks
    if len(desc) > 300:
        desc = desc[:300] + "..."
    skills = [s.name for s in agent.content.skills]
    if skills:
        desc += f"\nSkills: {', '.join(skills)}"
    return desc


def _make_ask_specialist(remote_url: str, agent_name: str):
    """Factory that creates the ask_specialist tool function for a manifest agent.

    Uses a factory to properly capture closure variables per agent (avoids
    the loop closure pitfall where all functions share the last iteration's values).
    """
    _remote_agent = None

    async def ask_specialist(message: str) -> str:
        """Send a work request to the remote specialist and get their response.

        Provide a clear, detailed message describing what needs to be done.
        The specialist will perform the work and return their result.
        """
        nonlocal _remote_agent
        try:
            # Lazy import — A2A SDK is optional
            from autogen.a2a import A2aRemoteAgent

            if _remote_agent is None:
                _remote_agent = A2aRemoteAgent(url=remote_url, name=f"remote-{agent_name}")
                logger.info(f"Created A2A remote agent for {agent_name} at {remote_url}")

            success, reply = await _remote_agent.a_generate_remote_reply(
                messages=[{"role": "user", "content": message}],
            )
            if reply:
                content = reply.get("content", "")
                if content:
                    return content
                return "Specialist returned an empty response."
            return "Specialist did not respond."
        except ImportError:
            return "Error: A2A SDK not installed. Install with: pip install ag2[a2a]"
        except Exception as e:
            return f"Error communicating with {agent_name} specialist: {e}"

    return ask_specialist


def _build_a2a_workers(
    agents: list[ManifestAgent],
    worker_llm_config: Any,
    agent_names: list[str] | None,
) -> list[ConversableAgent]:
    """Build wrapper ConversableAgents for remote A2A agents."""
    workers = []

    for agent_def in agents:
        if agent_names and agent_def.name not in agent_names:
            continue

        description = _make_agent_description(agent_def)

        # Build a system message that tells the LLM to delegate work
        skills_text = ""
        if agent_def.content.skills:
            skills_list = "\n".join(f"  - {s.name}: {s.description}" for s in agent_def.content.skills)
            skills_text = f"\nSpecialist skills:\n{skills_list}\n"

        system_message = (
            f"You are a task executor that delegates work to the {agent_def.name} specialist.\n"
            f"{skills_text}\n"
            "When given a task to work on:\n"
            "1. If the task is not yet assigned to you, claim it using claim_task\n"
            "2. Delegate the actual work to the specialist using ask_specialist "
            "with a clear, detailed message describing what needs to be done\n"
            "3. Complete the task using complete_task with the specialist's response "
            "as the result\n\n"
            "IMPORTANT: Always delegate work to the specialist using ask_specialist. "
            "Do NOT try to do the work yourself."
        )

        worker = ConversableAgent(
            name=agent_def.name,
            llm_config=worker_llm_config,
            system_message=system_message,
            description=description,
        )

        # Register the delegate tool using the factory to capture closure vars
        func = _make_ask_specialist(agent_def.url, agent_def.name)

        # register_for_execution stores in _function_map (must be first)
        worker.register_for_execution()(func)
        # register_for_llm generates JSON schema and updates llm_config
        worker.register_for_llm(
            description=(
                f"Send a work request to the {agent_def.name} specialist and get their response. "
                "Provide a clear, detailed message describing what needs to be done."
            )
        )(func)

        # Store manifest metadata on the agent for introspection
        worker._manifest_agent = agent_def  # type: ignore[attr-defined]

        workers.append(worker)
        logger.info(f"Created A2A manifest worker: {agent_def.name} -> {agent_def.url}")

    return workers


def _build_local_workers(
    agents: list[LocalAgentDef],
    worker_llm_config: Any,
    agent_names: list[str] | None,
) -> list[ConversableAgent]:
    """Build ConversableAgents from inline local agent definitions."""
    workers = []

    for agent_def in agents:
        if agent_names and agent_def.name not in agent_names:
            continue

        llm_config = _merge_llm_config(worker_llm_config, agent_def.llm_config or None)

        worker = ConversableAgent(
            name=agent_def.name,
            llm_config=llm_config,
            system_message=agent_def.system_message or None,
            description=agent_def.description,
        )

        workers.append(worker)
        logger.info(f"Created local manifest worker: {agent_def.name}")

    return workers


def _build_reference_workers(
    agents: list[ReferenceAgentDef],
    worker_llm_config: Any,
    agent_names: list[str] | None,
) -> list[ConversableAgent]:
    """Build AG2 built-in agents from reference definitions."""
    workers = []

    for agent_def in agents:
        if agent_names and agent_def.name not in agent_names:
            continue

        cls = _import_agent_class(agent_def.agent_class)

        # Expand env vars in constructor args
        expanded_args = _expand_env_vars(agent_def.args)

        # Build LLM config with per-agent overrides
        llm_config = _merge_llm_config(worker_llm_config, agent_def.llm_config or None)

        # Construct the agent
        kwargs: dict[str, Any] = {
            "name": agent_def.name,
            "llm_config": llm_config,
            **expanded_args,
        }
        if agent_def.description:
            kwargs["description"] = agent_def.description

        worker = cls(**kwargs)
        workers.append(worker)
        logger.info(f"Created reference manifest worker: {agent_def.name} (class={agent_def.agent_class})")

    return workers


def build_workers_from_manifest(
    manifest: AgentManifest,
    worker_llm_config: Any,
    *,
    agent_names: list[str] | None = None,
) -> list[ConversableAgent]:
    """Create worker agents from all three manifest categories.

    Processes agents from three sources:

    1. **Remote A2A** (``a2a`` + legacy ``agents``): Wrapper ConversableAgents
       with ``ask_specialist`` tool for delegation to remote A2A endpoints.

    2. **Local** (``local_agents``): Regular ConversableAgents with inline
       system_message and optional model/max_tokens overrides.

    3. **Reference** (``reference_agents``): AG2 built-in agent classes
       imported by name, with env var expansion in constructor args.

    Args:
        manifest: Parsed agent manifest.
        worker_llm_config: Base LLM config for worker agents. Per-agent
            ``model`` and ``max_tokens`` overrides are applied on top.
        agent_names: Optional list of agent names to include. If None,
            includes all agents from all categories.

    Returns:
        List of agents ready to add to a Team.
    """
    workers: list[ConversableAgent] = []

    # 1. Remote A2A — merge legacy `agents` with `a2a`
    all_a2a = list(manifest.a2a) + list(manifest.agents)
    workers.extend(_build_a2a_workers(all_a2a, worker_llm_config, agent_names))

    # 2. Local agents
    workers.extend(_build_local_workers(manifest.local_agents, worker_llm_config, agent_names))

    # 3. Reference agents
    workers.extend(_build_reference_workers(manifest.reference_agents, worker_llm_config, agent_names))

    return workers
