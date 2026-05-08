# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypedDict

import httpx
from a2a.client import ClientCallInterceptor
from a2a.types import AgentCard
from typing_extensions import Self, Unpack

from autogen.beta.config.config import ModelConfig

from .client import A2AClient
from .errors import A2AInvalidCardError
from .transports import TransportName

if TYPE_CHECKING:
    import grpc.aio


class A2AConfigOverrides(TypedDict, total=False):
    url: str
    transports: Sequence[TransportName]
    streaming: bool
    headers: Mapping[str, str] | None
    timeout: float | None
    max_reconnects: int
    reconnect_backoff: float
    polling_interval: float
    input_required_timeout: float | None
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None
    interceptors: Sequence[ClientCallInterceptor]
    grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None
    tenant: str | None
    history_length: int | None


@dataclass(slots=True)
class A2AConfig(ModelConfig):
    """Connection config for an A2A agent acting as an LLM provider.

    ``url`` is the base address of the remote A2A server (the agent card
    is fetched from ``{url}/.well-known/agent-card.json`` per spec).

    ``transports`` is the ordered preference list of protocol bindings
    the client is willing to negotiate. The SDK picks the first one the
    server card declares as supported. Default ``("jsonrpc",)``.

    ``polling_interval`` is used when the server card declares
    ``capabilities.streaming=False`` or when the user opts into
    ``streaming=False``: ``Task`` state is polled via ``get_task`` every
    ``polling_interval`` seconds until terminal.

    ``input_required_timeout`` caps how long the client waits on the
    HITL hook when the server transitions a task into
    ``TASK_STATE_INPUT_REQUIRED``. ``None`` means wait indefinitely
    (matches ``ConversationContext.input``).

    ``grpc_channel_factory`` builds a ``grpc.aio.Channel`` for a given
    URL when the negotiated transport is gRPC. Required only if
    ``"grpc"`` is in ``transports`` and the server actually picks it.

    ``tenant`` scopes every outgoing request to a specific tenant on the
    remote server (A2A multi-tenancy: a single shared backend can isolate
    data per tenant). Per-call override is available via
    ``context.variables["a2a:tenant"]``.

    ``history_length`` truncates the server-side ``Task.history`` echoed
    back on ``get_task`` / list operations to the most recent N messages.
    Pure server-side hint — does not change what the client uploads.
    """

    url: str
    transports: Sequence[TransportName] = ("jsonrpc",)
    streaming: bool = True
    headers: Mapping[str, str] | None = None
    timeout: float | None = 60.0
    max_reconnects: int = 3
    reconnect_backoff: float = 0.5
    polling_interval: float = 0.5
    input_required_timeout: float | None = None
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None = field(default=None, repr=False)
    interceptors: Sequence[ClientCallInterceptor] = ()
    grpc_channel_factory: Callable[[str], "grpc.aio.Channel"] | None = field(default=None, repr=False)
    preset_card: AgentCard | None = field(default=None, repr=False)
    tenant: str | None = None
    history_length: int | None = None

    def copy(self, /, **overrides: Unpack[A2AConfigOverrides]) -> Self:
        return replace(self, **overrides)

    @classmethod
    def from_card(
        cls,
        card: AgentCard,
        *,
        url: str | None = None,
        **overrides: Any,
    ) -> Self:
        """Construct a config from a pre-fetched ``AgentCard``.

        Useful when the card has already been resolved (e.g. via a
        discovery service) and a network round-trip on connect can be
        skipped. ``url`` defaults to the first interface declared on
        the card; raises ``ValueError`` if neither is available.
        """
        resolved_url = url or _first_interface_url(card)
        if not resolved_url:
            raise A2AInvalidCardError(
                "AgentCard has no supported_interfaces and no `url` override was provided",
            )
        return cls(url=resolved_url, preset_card=card, **overrides)

    def create(self) -> A2AClient:
        return A2AClient(
            url=self.url,
            transports=tuple(self.transports),
            streaming=self.streaming,
            headers=dict(self.headers) if self.headers else None,
            timeout=self.timeout,
            max_reconnects=self.max_reconnects,
            reconnect_backoff=self.reconnect_backoff,
            polling_interval=self.polling_interval,
            input_required_timeout=self.input_required_timeout,
            httpx_client_factory=self.httpx_client_factory,
            interceptors=tuple(self.interceptors),
            grpc_channel_factory=self.grpc_channel_factory,
            preset_card=self.preset_card,
            tenant=self.tenant,
            history_length=self.history_length,
        )


def _first_interface_url(card: AgentCard) -> str | None:
    interfaces = card.supported_interfaces
    if not interfaces:
        return None
    return interfaces[0].url or None
