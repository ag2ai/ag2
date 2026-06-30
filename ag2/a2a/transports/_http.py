# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import httpx
from a2a.client import Client, ClientCallInterceptor, ClientConfig, ClientFactory
from a2a.client.client_factory import TransportProtocol
from a2a.types import AgentCard, AgentInterface
from a2a.utils.constants import PROTOCOL_VERSION_1_0
from packaging.version import InvalidVersion, Version

from ..errors import A2AIncompatibleProtocolVersionError, A2AInvalidCardError
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


def select_interface(
    card: AgentCard, *, url: str, prefer: TransportName | None
) -> tuple[AgentInterface, TransportName]:
    """Pick the ``AgentInterface`` AG2 will connect through, with its transport.

    Resolution: 1) ``prefer`` matches a declared binding (raise if absent);
    2) interface whose ``url`` matches ``url``; 3) first listed interface.

    Returning the interface itself (not just the transport name) lets the
    caller validate the *exact* interface that will be used — when a card
    declares two interfaces with the same binding, the first-by-binding one
    is not necessarily the one ``url`` resolution would select.
    """
    interfaces = list(card.supported_interfaces)
    if not interfaces:
        raise A2AInvalidCardError(f"AgentCard at {url!r} has no supported_interfaces")

    if prefer is not None:
        for iface in interfaces:
            if binding_to_transport(iface.protocol_binding) == prefer:
                return iface, prefer
        raise A2AInvalidCardError(
            f"AgentCard at {url!r} does not declare prefer={prefer!r}; "
            f"available: {[iface.protocol_binding for iface in interfaces]}",
        )

    for iface in interfaces:
        if iface.url == url:
            transport = binding_to_transport(iface.protocol_binding)
            if transport is not None:
                return iface, transport

    first = interfaces[0]
    transport = binding_to_transport(first.protocol_binding)
    if transport is None:
        raise A2AInvalidCardError(
            f"AgentCard at {url!r} declares unsupported binding {first.protocol_binding!r}",
        )
    return first, transport


def validate_protocol_version(iface: AgentInterface, *, url: str, transport: TransportName) -> None:
    """Raise ``A2AIncompatibleProtocolVersionError`` if ``iface`` advertises an
    A2A protocol version older than 1.0.

    A2A v1.0 is a breaking change vs. the 0.x
    (see https://a2a-protocol.org/latest/announcing-1.0/), so we refuse pre-1.0
    interfaces at connect time rather than letting them surface as obscure
    RPC failures later.

    An empty / missing / unparsable ``protocol_version`` is treated as
    compatible: the field is optional and the A2A SDK defaults it to the
    current version, so rejecting it would break spec-conforming 1.0 servers.
    """
    raw = iface.protocol_version
    if not raw:
        return
    try:
        version = Version(raw)
    except InvalidVersion:
        return
    if version < Version(PROTOCOL_VERSION_1_0):
        raise A2AIncompatibleProtocolVersionError(url=url, transport=transport, protocol_version=raw)


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
