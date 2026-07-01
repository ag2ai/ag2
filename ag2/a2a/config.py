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

from ag2.config.config import ModelConfig

from .client import A2AClient
from .errors import A2AInvalidCardError
from .transports import TransportName

if TYPE_CHECKING:
    import grpc.aio


class A2AConfigOverrides(TypedDict, total=False):
    card_url: str
    prefer: TransportName | None
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
    preset_card: AgentCard | None
    tenant: str | None
    history_length: int | None


@dataclass(slots=True)
class A2AConfig(ModelConfig):
    """Connection config for an A2A agent acting as an LLM provider.

    ``card_url`` is the HTTP(S) URL where the agent card is published
    (fetched from ``{card_url}/.well-known/agent-card.json`` per spec).
    The actual transport endpoint for the request/response exchange is
    read from ``card.supported_interfaces`` — the user does not pass it.

    ``prefer`` selects a transport when the server card declares more
    than one binding. ``None`` (default) auto-picks: if exactly one
    interface matches ``card_url`` it is used; otherwise the first
    server-listed interface wins. Pass ``"jsonrpc" | "rest" | "grpc"``
    to force a specific binding.

    ``polling_interval`` is used when the server card declares
    ``capabilities.streaming=False`` or when the user opts into
    ``streaming=False``: ``Task`` state is polled via ``get_task`` every
    ``polling_interval`` seconds until terminal.

    ``input_required_timeout`` caps how long the client waits on the
    HITL hook when the server transitions a task into
    ``TASK_STATE_INPUT_REQUIRED``. ``None`` means wait indefinitely
    (matches ``ConversationContext.input``).

    ``grpc_channel_factory`` builds a ``grpc.aio.Channel`` for a given
    URL when the resolved transport is gRPC. Optional — defaults to
    insecure_channel via ``default_grpc_channel_factory``.

    ``tenant`` scopes every outgoing request to a specific tenant on the
    remote server (A2A multi-tenancy: a single shared backend can isolate
    data per tenant). Per-call override is available via
    ``context.variables["a2a:tenant"]``.

    ``history_length`` truncates the server-side ``Task.history`` echoed
    back on ``get_task`` / list operations to the most recent N messages.
    Pure server-side hint — does not change what the client uploads.
    """

    card_url: str
    prefer: TransportName | None = None
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
        card_url: str | None = None,
        **overrides: Any,
    ) -> Self:
        """Construct a config from a pre-fetched ``AgentCard``.

        Useful when the card has already been resolved (e.g. via a
        discovery service) and a network round-trip on connect can be
        skipped. ``card_url`` defaults to the first interface declared
        on the card; raises ``A2AInvalidCardError`` if neither is
        available.
        """
        resolved_url = card_url or _first_interface_url(card)
        if not resolved_url:
            raise A2AInvalidCardError(
                "AgentCard has no supported_interfaces and no `card_url` override was provided",
            )
        return cls(card_url=resolved_url, preset_card=card, **overrides)

    def create(self) -> A2AClient:
        return A2AClient(
            card_url=self.card_url,
            prefer=self.prefer,
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
    return next((i.url for i in card.supported_interfaces if i.url), None)
