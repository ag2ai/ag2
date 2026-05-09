# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT

from autogen.beta.agent import Agent

from .extension import EXTENSION_URI
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
    """
    if "grpc" in transports and grpc_url is None:
        raise ValueError("grpc_url is required when 'grpc' is in transports")

    description_text = description or _agent_description(agent)
    skills = [
        AgentSkill(
            id=agent.name,
            name=agent.name,
            description=description_text or agent.name,
            tags=[],
        )
    ]
    capabilities = AgentCapabilities(
        streaming=True,
        push_notifications=False,
        extensions=[
            AgentExtension(
                uri=EXTENSION_URI,
                description="AG2 client-side tool execution",
                required=False,
            ),
        ],
    )
    return AgentCard(
        name=agent.name,
        description=description_text,
        version=version,
        default_input_modes=list(_DEFAULT_INPUT_MODES),
        default_output_modes=list(_DEFAULT_OUTPUT_MODES),
        capabilities=capabilities,
        skills=skills,
        supported_interfaces=_build_interfaces(
            transports=transports,
            url=url,
            rest_url=rest_url,
            rest_path_prefix=rest_path_prefix,
            grpc_url=grpc_url,
        ),
    )


def _build_interfaces(
    *,
    transports: Sequence[TransportName],
    url: str,
    rest_url: str | None,
    rest_path_prefix: str,
    grpc_url: str | None,
) -> list[AgentInterface]:
    interfaces: list[AgentInterface] = []
    for name in transports:
        if name == "jsonrpc":
            interfaces.append(
                AgentInterface(
                    url=url,
                    protocol_binding=TransportProtocol.JSONRPC.value,
                    protocol_version=PROTOCOL_VERSION_CURRENT,
                ),
            )
        elif name == "rest":
            interfaces.append(
                AgentInterface(
                    url=rest_url if rest_url is not None else url + rest_path_prefix,
                    protocol_binding=TransportProtocol.HTTP_JSON.value,
                    protocol_version=PROTOCOL_VERSION_CURRENT,
                ),
            )
        elif name == "grpc":
            assert grpc_url is not None  # validated above
            interfaces.append(
                AgentInterface(
                    url=grpc_url,
                    protocol_binding=TransportProtocol.GRPC.value,
                    protocol_version=PROTOCOL_VERSION_CURRENT,
                ),
            )
    return interfaces


def _agent_description(agent: Agent) -> str:
    prompt = agent._system_prompt if agent._system_prompt else None
    if prompt:
        return prompt[0]
    return ""
