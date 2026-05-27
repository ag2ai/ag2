# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import httpx
from a2a.client import Client, ClientCallInterceptor, ClientConfig, ClientFactory
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCard

from ..errors import A2AInvalidCardError
from . import TransportName
from .grpc import default_grpc_channel_factory

if TYPE_CHECKING:
    import grpc.aio

# Short transport names ↔ SDK protocol-binding strings (used in
# ``ClientConfig.supported_protocol_bindings`` and ``AgentInterface.protocol_binding``).
_TRANSPORT_BINDINGS: dict[str, str] = {
    "jsonrpc": TransportProtocol.JSONRPC.value,
    "rest": TransportProtocol.HTTP_JSON.value,
    "grpc": TransportProtocol.GRPC.value,
}

_BINDING_TO_TRANSPORT: dict[str, TransportName] = {v: k for k, v in _TRANSPORT_BINDINGS.items()}  # type: ignore[misc]


def binding_to_transport(binding: str) -> TransportName | None:
    """SDK protocol-binding string → our short transport name (``None`` if unsupported)."""
    return _BINDING_TO_TRANSPORT.get(binding)


def select_transport(card: AgentCard, *, url: str, prefer: TransportName | None) -> TransportName:
    """Pick a transport from ``card.supported_interfaces``.

    Resolution: 1) ``prefer`` matches a declared binding (raise if absent);
    2) interface whose ``url`` matches ``url``; 3) first listed interface.
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

    A user-supplied ``factory`` owns the client entirely; we do not
    mutate its headers (the factory may return a shared instance).
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


def make_a2a_client(
    *,
    card: AgentCard,
    httpx_client: httpx.AsyncClient,
    streaming: bool,
    transport: TransportName,
    interceptors: Sequence[ClientCallInterceptor] = (),
    grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None = None,
) -> Client:
    """Build an A2A SDK ``Client`` for the resolved transport.

    The SDK factory negotiates streaming vs. polling automatically based on
    ``card.capabilities.streaming`` and ``ClientConfig.streaming``.
    Importing ``default_grpc_channel_factory`` lazily would keep HTTP-only
    deployments from pulling ``grpcio`` — currently eager since the cycle
    avoidance is handled via ``_common.py``.
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
    return ClientFactory(config).create(card, interceptors=list(interceptors) if interceptors else None)
