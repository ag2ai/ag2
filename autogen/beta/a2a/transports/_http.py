# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import httpx
from a2a.client import A2ACardResolver, Client, ClientCallInterceptor, ClientConfig, ClientFactory
from a2a.client.client_factory import TransportProtocol
from a2a.client.errors import AgentCardResolutionError
from a2a.server.context import ServerCallContext
from a2a.types import AgentCard

from ..errors import A2AInvalidCardError
from . import TransportName
from .grpc import default_grpc_channel_factory

if TYPE_CHECKING:
    import grpc.aio

CardModifier: TypeAlias = Callable[[AgentCard], Awaitable[AgentCard]]
ExtendedCardModifier: TypeAlias = Callable[[AgentCard, ServerCallContext], Awaitable[AgentCard]]

# Legacy A2A v0.x agent-card path. SDK 1.x dropped this constant; we keep
# it for backward compatibility with old clients/servers that still rely
# on the pre-v1 well-known path.
LEGACY_AGENT_CARD_PATH = "/.well-known/agent.json"
DEFAULT_AGENT_CARD_PATH = "/.well-known/agent-card.json"

# Maps our short transport names to the SDK's protocol-binding strings.
# The SDK uses these strings both in ``ClientConfig.supported_protocol_bindings``
# and in ``AgentInterface.protocol_binding`` on the agent card.
_TRANSPORT_BINDINGS: dict[str, str] = {
    "jsonrpc": TransportProtocol.JSONRPC.value,
    "rest": TransportProtocol.HTTP_JSON.value,
    "grpc": TransportProtocol.GRPC.value,
}

_BINDING_TO_TRANSPORT: dict[str, TransportName] = {v: k for k, v in _TRANSPORT_BINDINGS.items()}  # type: ignore[misc]


def binding_to_transport(binding: str) -> TransportName | None:
    """Map an SDK protocol-binding string back to our short transport name.

    Returns ``None`` if the binding is not one we support — caller can
    raise a meaningful error then.
    """
    return _BINDING_TO_TRANSPORT.get(binding)


def select_transport(card: AgentCard, *, url: str, prefer: TransportName | None) -> TransportName:
    """Pick a transport from ``card.supported_interfaces``.

    Resolution order:
    1. ``prefer`` set → match it against a declared binding; raise if absent.
    2. Otherwise → prefer the interface whose ``url`` matches ``url``
       (the common case: one URL, one binding).
    3. Fallback → first server-listed interface.
    """
    interfaces = list(card.supported_interfaces)
    if not interfaces:
        raise A2AInvalidCardError(f"AgentCard at {url!r} has no supported_interfaces")

    if prefer is not None:
        for iface in interfaces:
            transport = binding_to_transport(iface.protocol_binding)
            if transport == prefer:
                return transport
        raise A2AInvalidCardError(
            f"AgentCard at {url!r} does not declare prefer={prefer!r}; "
            f"available: {[iface.protocol_binding for iface in interfaces]}",
        )

    for iface in interfaces:
        if iface.url == url:
            transport = binding_to_transport(iface.protocol_binding)
            if transport is not None:
                return transport

    first = interfaces[0]
    transport = binding_to_transport(first.protocol_binding)
    if transport is None:
        raise A2AInvalidCardError(
            f"AgentCard at {url!r} declares unsupported binding {first.protocol_binding!r}",
        )
    return transport


def make_httpx_client(
    *,
    headers: Mapping[str, str] | None,
    timeout: float | None,
    factory: Callable[[], httpx.AsyncClient] | None,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` for talking to an A2A server.

    When a ``factory`` is supplied it owns the client entirely — we do not
    mutate its headers, since the factory may return a shared instance and
    leaking caller-specific headers into it would contaminate other
    requests. If you need extra headers on top of a factory-built client,
    set them inside the factory.
    """
    if factory is not None:
        if headers:
            warnings.warn(
                "`headers` is ignored when `httpx_client_factory` is provided; "
                "set headers on the client returned by the factory instead.",
                RuntimeWarning,
                stacklevel=2,
            )
        return factory()
    return httpx.AsyncClient(headers=dict(headers) if headers else None, timeout=timeout)


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
    transport: TransportName,
    interceptors: Sequence[ClientCallInterceptor] = (),
    grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None = None,
) -> Client:
    """Build an A2A SDK ``Client`` for the given resolved transport.

    ``transport`` is already a single, resolved binding — selection
    happens upstream by matching the card_url against
    ``card.supported_interfaces`` (with optional ``prefer`` override).
    The factory negotiates streaming vs. polling automatically based on
    ``card.capabilities.streaming`` and ``ClientConfig.streaming``.

    When ``transport == "grpc"`` and ``grpc_channel_factory`` is not
    supplied, the insecure default from ``transports.grpc`` is
    auto-attached. Importing it lazily here keeps HTTP-only deployments
    from pulling ``grpcio`` at import time.
    """
    if transport == "grpc" and grpc_channel_factory is None:
        grpc_channel_factory = default_grpc_channel_factory

    config = ClientConfig(
        streaming=streaming and card.capabilities.streaming,
        polling=not (streaming and card.capabilities.streaming),
        httpx_client=httpx_client,
        supported_protocol_bindings=[_TRANSPORT_BINDINGS[transport]],
        grpc_channel_factory=grpc_channel_factory,
    )
    factory = ClientFactory(config)
    return factory.create(card, interceptors=list(interceptors) if interceptors else None)
