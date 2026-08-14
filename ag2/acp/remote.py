# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Drive an ACP agent that is already running somewhere else.

:class:`ACPRemoteConfig` is an :class:`~.config.ACPConfig` pointed at a URL
instead of a command, and :func:`open_remote_connection` is a third
implementation of the connection hook the launch-based config already uses — it
yields a connection and no process handle. Nothing above that hook learns that
transports exist.

The workspace stays on the AG2 side. ACP's ``fs/*`` and terminal methods are
requests *from* the agent *to* the client, and AG2 is the client, so what moves
off-host is the agent's reasoning, not its working copy: a remote agent reads
and writes the local workspace through AG2's root-confined mediation, and its
terminal commands run where that workspace is.

This module needs the transports' own dependencies (an HTTP/2 client and a
WebSocket library), which ship with ``agent-client-protocol[http]`` rather than
with ``ag2[acp]`` — most ACP users drive a local subprocess and should not pay
for a WebSocket stack. A broken install surfaces on
``from ag2.acp import ACPRemoteConfig`` with that install hint.
"""

from asyncio.subprocess import Process
from collections.abc import AsyncGenerator, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import acp
import httpx
import websockets
from acp.core import ClientSideConnection
from acp.http.client import AcpHttpStatusError, create_http_stream
from acp.ws.client import create_websocket_stream

from .config import ACPConfig, _dispatch_kwargs
from .tool_gateway import GatewayAddress, MCPCapabilityError
from .transport import ACPTransport, resolve_transport

if TYPE_CHECKING:
    from acp._transport import Transport

__all__ = ("ACPRemoteConfig", "open_remote_connection")


@dataclass(slots=True, kw_only=True)
class ACPRemoteConfig(ACPConfig):
    """Drive a CLI coding agent reachable at a URL.

    An :class:`~ag2.acp.config.ACPConfig` that dials instead of launching: the
    workspace, model, policies and timeouts are inherited and mean exactly what
    they mean locally, and the launch-only fields are taken out of the
    constructor rather than left as arguments that would do nothing. Passing
    ``command=`` or ``env=`` here is a ``TypeError``, so a command and a URL can
    never disagree about how to reach the agent.

    Its own fields are keyword-only. Nothing was ever constructed positionally
    here — the class is new — so keyword-only costs nothing and means a field
    added between two others cannot shift a call.

    Attributes:
        url: Where the agent speaks ACP, e.g. ``https://box.internal/acp`` or
            ``wss://box.internal/acp``. The scheme picks the transport.
        headers: Sent with every request on the connection — this is how a
            bearer-token gateway is reached. Source the values the same careful
            way the subprocess environment is; a token committed to a config
            literal is a token leaked. Kept out of ``repr`` for the same reason:
            a config logged or echoed in a traceback must not carry the token
            with it.
        transport: Forces the transport regardless of the URL's scheme, for a
            proxy or gateway that does not follow the conventions. ``None``
            (default) infers it from the scheme.
        gateway_address: The ``host`` or ``host:port`` a remote agent should
            dial to reach AG2's tool gateway, for a caller who has arranged
            network reachability between the two. Supplying it also widens the
            gateway's bind address, DNS-rebinding allowlist and allowed origins
            to match. Required when ``expose_tools`` is on: the gateway's
            default loopback binding is unreachable from another host, and
            handing a remote agent a loopback URL would leave it silently
            toolless. It is never inferred from the presence of a URL — opening
            a local port to the network is the caller's decision to state.
            Note what that decision buys: the gateway serves plain HTTP whose
            only credential is an unguessable path segment, so keep the address
            on a network only the agent can reach.
    """

    # Inherited, then withdrawn from the constructor: ``init=False`` is what
    # makes ``ACPRemoteConfig(url=..., command=[...])`` a TypeError instead of an
    # argument that quietly launches nothing. They keep their empty defaults, so
    # the inherited code that reads them — ``_agent_label`` — still works.
    command: list[str] = field(init=False, repr=False, default_factory=list)
    env: dict[str, str] | None = field(init=False, repr=False, default=None)

    url: str
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    transport: ACPTransport | None = None
    gateway_address: str | None = None

    # A dropped SSE stream or closed socket reaches the caller as
    # ``ConnectionError`` (the SDK rejects pending requests with one); a POST
    # rejected before any JSON-RPC response exists surfaces as the SDK's own
    # status error or an httpx/websockets failure.
    _transport_errors: ClassVar[tuple[type[BaseException], ...]] = (
        ConnectionError,
        AcpHttpStatusError,
        httpx.HTTPError,
        websockets.WebSocketException,
    )

    def __post_init__(self) -> None:
        # Both fail at construction rather than at session start: a typo in
        # either surfaces a long way from where it was written otherwise.
        resolve_transport(self.url, self.transport)
        if self.gateway_address is not None:
            GatewayAddress.parse(self.gateway_address)

    @property
    def _transport_label(self) -> str:
        return resolve_transport(self.url, self.transport)

    @property
    def _agent_label(self) -> str:
        return self.url

    def _connect_transport(
        self, client: acp.Client
    ) -> AbstractAsyncContextManager[tuple[ClientSideConnection, Process | None]]:
        return open_remote_connection(
            client,
            url=self.url,
            transport=resolve_transport(self.url, self.transport),
            headers=self.headers,
        )

    def _gateway_address(self) -> GatewayAddress:
        if self.gateway_address is None:
            raise MCPCapabilityError(
                self.url,
                reason=(
                    f"AG2 cannot expose the agent's tools to the remote ACP agent at {self.url!r}: the "
                    "tool gateway binds loopback only, so the agent has no address it could reach it on. "
                    "Set gateway_address='host:port' to the address the agent should dial (which also "
                    "binds the gateway there), or set expose_tools=False."
                ),
            )
        return GatewayAddress.parse(self.gateway_address)


@asynccontextmanager
async def open_remote_connection(
    client: acp.Client,
    *,
    url: str,
    transport: ACPTransport,
    headers: Mapping[str, str] | None = None,
) -> AsyncGenerator[tuple[ClientSideConnection, Process | None]]:
    """Open an ACP connection to a remote agent; yield it with no process handle.

    The third implementation of the connection hook, alongside the subprocess
    spawn and the in-process test double. Closing it closes the transport —
    there is nothing to kill, and nothing to reconnect: a dropped connection
    fails the turn and the session.

    Args:
        client: The AG2-side ACP client the connection is bound to.
        url: The remote agent's ACP endpoint.
        transport: Which transport to open, already resolved.
        headers: Extra headers to send, e.g. ``Authorization``.
    """
    sent_headers = dict(headers) if headers else None
    stream: Transport = (
        create_http_stream(url, headers=sent_headers)
        if transport == "http"
        else await create_websocket_stream(url, headers=sent_headers)
    )
    # `use_unstable_protocol` registers the `elicitation/*` client routes, as on
    # the subprocess path; without it the SDK answers the agent's question with
    # method-not-found before the bridge ever sees it.
    conn = acp.connect_to_agent(client, stream, use_unstable_protocol=True, **_dispatch_kwargs(client))
    try:
        yield conn, None
    finally:
        await conn.close()
