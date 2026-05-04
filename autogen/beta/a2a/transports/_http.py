# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallInterceptor, ClientConfig, ClientFactory
from a2a.client.client_factory import TransportProtocol
from a2a.client.errors import AgentCardResolutionError
from a2a.types import AgentCard

from .grpc import default_grpc_channel_factory

if TYPE_CHECKING:
    from a2a.server.context import ServerCallContext

CardModifier: TypeAlias = Callable[[AgentCard], Awaitable[AgentCard]]
ExtendedCardModifier: TypeAlias = Callable[[AgentCard, "ServerCallContext"], Awaitable[AgentCard]]

# Legacy A2A v0.x agent-card path. SDK 1.x dropped this constant; we keep
# it for backward compatibility with old clients/servers that still rely
# on the pre-v1 well-known path.
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"
DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"

TransportName = Literal["jsonrpc", "rest", "grpc"]

# Maps our short transport names to the SDK's protocol-binding strings.
# The SDK uses these strings both in ``ClientConfig.supported_protocol_bindings``
# and in ``AgentInterface.protocol_binding`` on the agent card.
_TRANSPORT_BINDINGS: dict[str, str] = {
    "jsonrpc": TransportProtocol.JSONRPC.value,
    "rest": TransportProtocol.HTTP_JSON.value,
    "grpc": TransportProtocol.GRPC.value,
}


def make_httpx_client(
    *,
    headers: Mapping[str, str] | None,
    timeout: float | None,
    factory: Callable[[], httpx.AsyncClient] | None,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` for talking to an A2A server.

    A user-supplied ``factory`` wins over our defaults (useful for tests
    via ``httpx.ASGITransport`` and for custom auth flows).
    """
    client = factory() if factory is not None else httpx.AsyncClient(timeout=timeout)
    if headers:
        for k, v in headers.items():
            client.headers[k] = v
    return client


async def fetch_card(
    httpx_client: httpx.AsyncClient,
    *,
    url: str,
) -> AgentCard:
    """Fetch the agent card with a legacy-path fallback.

    Tries the SDK 1.x default ``/.well-known/agent-card.json`` first.
    If the server returns 404, falls back to the v0.x
    ``/.well-known/agent.json`` so we can still talk to legacy AG2
    deployments that haven't migrated.
    """
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
    try:
        return await resolver.get_agent_card()
    except AgentCardResolutionError as exc:
        if exc.status_code != 404:
            raise
        return await resolver.get_agent_card(relative_card_path=LEGACY_AGENT_CARD_PATH)


def make_a2a_client(
    *,
    card: AgentCard,
    httpx_client: httpx.AsyncClient,
    streaming: bool,
    transports: Sequence[TransportName] = ("jsonrpc",),
    interceptors: Sequence[ClientCallInterceptor] = (),
    grpc_channel_factory: Callable[[str], Any] | None = None,
) -> Client:
    """Build an A2A SDK ``Client`` honoring the requested transport preference.

    ``transports`` is an ordered preference list — the SDK picks the
    first one the server card declares as supported. The factory
    negotiates streaming vs. polling automatically based on
    ``card.capabilities.streaming`` and ``ClientConfig.streaming``.

    When ``"grpc"`` is in ``transports`` and ``grpc_channel_factory`` is
    not supplied, the insecure default from ``transports.grpc`` is
    auto-attached. Importing it lazily here keeps HTTP-only deployments
    from pulling ``grpcio`` at import time.
    """
    if "grpc" in transports and grpc_channel_factory is None:
        grpc_channel_factory = default_grpc_channel_factory

    bindings = [_TRANSPORT_BINDINGS[t] for t in transports]
    config = ClientConfig(
        httpx_client=httpx_client,
        streaming=streaming and card.capabilities.streaming,
        polling=not (streaming and card.capabilities.streaming),
        supported_protocol_bindings=bindings,
        grpc_channel_factory=grpc_channel_factory,
    )
    factory = ClientFactory(config)
    return factory.create(card, interceptors=list(interceptors) if interceptors else None)
