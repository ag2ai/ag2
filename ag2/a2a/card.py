# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence

from a2a.client.client_factory import TransportProtocol
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

from ag2.agent import Agent
from ag2.tools.skills.toolkit import SkillsToolkit

from .extension import EXTENSION_URI
from .security import Requirement, Scheme
from .transports import TransportName

_DEFAULT_VERSION = "1.0.0"
_DEFAULT_INPUT_MODES = ("text/plain", "application/json")
_DEFAULT_OUTPUT_MODES = ("text/plain", "application/json")


def build_card(
    agent: Agent,
    *,
    url: str,
    transports: Sequence[TransportName] = ("jsonrpc",),
    rest_url: str | None = None,
    rest_path_prefix: str = "",
    grpc_url: str | None = None,
    version: str = _DEFAULT_VERSION,
    description: str | None = None,
    push_notifications: bool = False,
    skills: Sequence[AgentSkill] | None = None,
    security: Sequence[Requirement] = (),
    provider: AgentProvider | None = None,
    documentation_url: str | None = None,
    icon_url: str | None = None,
    tenants: Mapping[TransportName, str] | None = None,
) -> AgentCard:
    """Construct an ``AgentCard`` describing an AG2 agent for A2A discovery.

    Always declares the ``urn:ag2:client-tools:v1`` extension as
    ``required=False`` — the server can transparently fall back to a
    plain text exchange when the client doesn't speak the extension.

    ``supported_interfaces`` is built from ``transports`` — one
    ``AgentInterface`` per enabled binding. JSON-RPC URL is ``url``;
    REST URL defaults to ``url + rest_path_prefix`` (same host:port,
    different path) but can be overridden via ``rest_url`` when REST
    lives on a different host:port; gRPC lives on its own ``grpc_url``.

    When ``skills`` is supplied, it replaces all auto-detection. When it
    is ``None``, ``build_card`` walks ``agent.tools`` for any
    :class:`SkillsToolkit` and publishes its ``agentskills.io``-style
    local skills as ``AgentSkill`` entries; if none are found, falls
    back to a single skill derived from ``agent.name`` /
    ``agent._system_prompt`` so the card stays spec-compliant.

    ``security`` is a sequence of :class:`Requirement` objects built via
    ``require(scheme, ...)``. Each entry is an AND-set of schemes; the
    list itself is OR-ed (any one requirement suffices). Underlying
    ``security_schemes`` on the card are auto-derived from the schemes
    referenced in ``security`` — no duplicate declarations needed.
    ``tenants`` maps a transport name to a tenant string surfaced on the
    corresponding ``AgentInterface.tenant``.
    """
    if "grpc" in transports and grpc_url is None:
        raise ValueError("grpc_url is required when 'grpc' is in transports")

    description_text = description or _agent_description(agent)
    resolved_skills = _resolve_skills(agent, skills, description_text)
    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=push_notifications,
        extensions=[
            AgentExtension(
                uri=EXTENSION_URI,
                description="AG2 client-side tool execution",
                required=False,
            ),
        ],
    )
    card_kwargs: dict[str, object] = {
        "name": agent.name,
        "description": description_text,
        "version": version,
        "default_input_modes": list(_DEFAULT_INPUT_MODES),
        "default_output_modes": list(_DEFAULT_OUTPUT_MODES),
        "capabilities": capabilities,
        "skills": resolved_skills,
        "supported_interfaces": _build_interfaces(
            transports=transports,
            url=url,
            rest_url=rest_url,
            rest_path_prefix=rest_path_prefix,
            grpc_url=grpc_url,
            tenants=tenants,
        ),
    }
    if security:
        seen: dict[str, Scheme] = {}
        for req in security:
            for scheme in req.schemes:
                seen[scheme.name] = scheme
        card_kwargs["security_schemes"] = {name: s.scheme for name, s in seen.items()}
        card_kwargs["security_requirements"] = [r.to_proto() for r in security]
    if provider is not None:
        card_kwargs["provider"] = provider
    if documentation_url is not None:
        card_kwargs["documentation_url"] = documentation_url
    if icon_url is not None:
        card_kwargs["icon_url"] = icon_url
    return AgentCard(**card_kwargs)


def _build_interfaces(
    *,
    transports: Sequence[TransportName],
    url: str,
    rest_url: str | None,
    rest_path_prefix: str,
    grpc_url: str | None,
    tenants: Mapping[TransportName, str] | None = None,
) -> list[AgentInterface]:
    tenant_map = tenants or {}
    interfaces: list[AgentInterface] = []
    for name in transports:
        if name == "jsonrpc":
            iface_url, protocol = url, TransportProtocol.JSONRPC.value
        elif name == "rest":
            iface_url = rest_url if rest_url is not None else url + rest_path_prefix
            protocol = TransportProtocol.HTTP_JSON.value
        else:
            assert name == "grpc" and grpc_url is not None  # validated above
            iface_url, protocol = grpc_url, TransportProtocol.GRPC.value
        interfaces.append(
            AgentInterface(
                url=iface_url,
                protocol_binding=protocol,
                protocol_version=PROTOCOL_VERSION_CURRENT,
                tenant=tenant_map.get(name, ""),
            ),
        )
    return interfaces


def _agent_description(agent: Agent) -> str:
    prompt = agent._system_prompt if agent._system_prompt else None
    if prompt:
        return prompt[0]
    return ""


def _resolve_skills(
    agent: Agent,
    explicit: Sequence[AgentSkill] | None,
    description: str,
) -> list[AgentSkill]:
    """Pick skills in priority order: explicit → auto-detected → default.

    Auto-detection walks ``agent.tools`` for any :class:`SkillsToolkit`
    and publishes its ``agentskills.io``-style local skills. The
    toolkit's own three tools (``list_skills`` / ``load_skill`` /
    ``run_skill_script``) are implementation detail and do not appear.
    """
    if explicit is not None:
        return list(explicit)

    auto = [
        AgentSkill(id=skill.name, name=skill.name, description=skill.metadata.description or skill.name)
        for tool in agent.tools
        if isinstance(tool, SkillsToolkit)
        for skill in tool.merged_skills()
    ]
    if auto:
        return auto

    return [
        AgentSkill(
            id=agent.name,
            name=agent.name,
            description=description or agent.name,
            tags=[],
        ),
    ]
