# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import httpx
from a2a.client import A2ACardResolver
from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentSkill
from a2a.utils.constants import TransportProtocol

from autogen.beta.tools.builtin.skills import SkillsTool
from autogen.beta.tools.skills import SkillsToolkit

from .mappers import CLIENT_TOOLS_EXTENSION_URI

if TYPE_CHECKING:
    from autogen.beta import Agent


async def fetch_card(httpx_client: httpx.AsyncClient, url: str) -> AgentCard:
    """Resolve the public ``AgentCard`` from ``/.well-known/agent-card.json``."""
    resolver = A2ACardResolver(httpx_client, url)
    return await resolver.get_agent_card()


def url_from_card(card: AgentCard, *, transport: str = TransportProtocol.JSONRPC.value) -> str:
    """Pick a transport URL out of an ``AgentCard``'s ``supported_interfaces``.

    A2A 1.0 dropped the single ``AgentCard.url`` field in favour of a list of
    ``AgentInterface(url, protocol_binding)``. Most callers want the JSON-RPC
    URL; pass ``transport=`` to pick another binding (REST, gRPC).
    Falls back to the first declared interface if the requested one is missing.
    """
    interfaces = list(card.supported_interfaces or ())
    if not interfaces:
        raise ValueError(f"AgentCard {card.name!r} declares no `supported_interfaces`")
    for iface in interfaces:
        if iface.protocol_binding == transport:
            return iface.url
    return interfaces[0].url


def build_card(
    agent: "Agent",
    *,
    url: str,
    rest_url: str | None = None,
    grpc_url: str | None = None,
    version: str = "0.1.0",
    description: str | None = None,
    capabilities: AgentCapabilities | None = None,
    default_input_modes: Sequence[str] | None = None,
    default_output_modes: Sequence[str] | None = None,
    extensions: Sequence[AgentExtension] | None = None,
    supports_extended: bool = False,
    supports_client_tools: bool = False,
) -> AgentCard:
    """Build a default ``AgentCard`` for an AG2 beta agent.

    ``url`` is treated as the JSON-RPC interface; pass ``rest_url`` and/or
    ``grpc_url`` to advertise additional transports.

    ``skills`` is always derived from the agent's installed
    ``SkillsTool`` / ``SkillsToolkit`` (see :func:`extract_skills`).

    ``supports_extended=True`` flips ``capabilities.extended_agent_card`` so
    A2A clients know to also fetch the authenticated extended card.
    """
    interfaces = list(_default_interfaces(url=url, rest_url=rest_url, grpc_url=grpc_url))
    base_caps = capabilities if capabilities is not None else AgentCapabilities(streaming=True)
    final_caps = _augment_capabilities(
        base_caps,
        extensions=extensions,
        include_client_tools=supports_client_tools,
        extended_agent_card=supports_extended,
    )
    return AgentCard(
        name=agent.name,
        description=description if description is not None else _default_description(agent),
        version=version,
        capabilities=final_caps,
        default_input_modes=list(default_input_modes) if default_input_modes is not None else ["text"],
        default_output_modes=list(default_output_modes) if default_output_modes is not None else ["text"],
        skills=extract_skills(agent),
        supported_interfaces=interfaces,
    )


def extract_skills(agent: "Agent") -> list[AgentSkill]:
    """Derive ``AgentCard.skills`` from skills tools installed on ``agent``.

    - ``SkillsToolkit`` (agentskills.io ``SKILL.md`` discovery) → one
      ``AgentSkill`` per discovered skill, using its name + description.
    - ``SkillsTool`` (provider-managed containers) → one ``AgentSkill``
      per declared id; description carries the version pin when set.
    """
    out: list[AgentSkill] = []
    for t in agent.tools:
        if isinstance(t, SkillsToolkit):
            # TODO(a2a-beta): replace `_runtime` access once SkillsToolkit exposes
            # a public discover() / installed-skills accessor.
            for meta in t._runtime.discover():
                out.append(
                    AgentSkill(
                        id=meta.name,
                        name=meta.name,
                        description=meta.description or f"Skill {meta.name}",
                    )
                )
        elif isinstance(t, SkillsTool):
            # TODO(a2a-beta): replace `_skills` access once SkillsTool exposes a
            # public iter-skills accessor.
            for s in t._skills:
                desc = f"Provider-managed skill {s.id}"
                if s.version:
                    desc += f" (v{s.version})"
                out.append(AgentSkill(id=s.id, name=s.id, description=desc))
    return out


def _default_interfaces(*, url: str, rest_url: str | None, grpc_url: str | None) -> Iterable[AgentInterface]:
    yield AgentInterface(url=url, protocol_binding=TransportProtocol.JSONRPC.value)
    if rest_url:
        yield AgentInterface(url=rest_url, protocol_binding=TransportProtocol.HTTP_JSON.value)
    if grpc_url:
        yield AgentInterface(url=grpc_url, protocol_binding=TransportProtocol.GRPC.value)


def _default_description(agent: "Agent") -> str:
    # TODO(a2a-beta): replace `_system_prompt` access once Agent exposes a
    # public system-prompt accessor.
    return "\n".join(agent._system_prompt) if agent._system_prompt else "AG2 beta agent"


def _augment_capabilities(
    caps: AgentCapabilities,
    *,
    extensions: Sequence[AgentExtension] | None,
    include_client_tools: bool,
    extended_agent_card: bool,
) -> AgentCapabilities:
    existing = list(caps.extensions or ())
    if extensions:
        existing.extend(extensions)
    if include_client_tools and not any(e.uri == CLIENT_TOOLS_EXTENSION_URI for e in existing):
        existing.append(
            AgentExtension(
                uri=CLIENT_TOOLS_EXTENSION_URI,
                description="AG2 client-side tools — server invokes tools declared on the client.",
            )
        )
    return AgentCapabilities(
        streaming=caps.streaming,
        push_notifications=caps.push_notifications,
        extensions=existing,
        extended_agent_card=extended_agent_card or caps.extended_agent_card,
    )
