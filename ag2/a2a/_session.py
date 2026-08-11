# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from a2a.client import A2ACardResolver, Client, ClientCallContext

from .config import A2AConfig
from .extension import extension_call_context, validate_extension_activation
from .transports._http import make_a2a_client, make_httpx_client, select_interface, validate_protocol_version


def with_tenant(config: A2AConfig, override: str | None, **kwargs: Any) -> dict[str, Any]:
    """Inject ``tenant`` into request kwargs from per-call override or config.

    Single source of truth for the tenant-resolution rule used by
    ``tasks``, ``push``, and (with a wrapping context-variables lookup)
    the ``A2AClient``. ``override`` wins over ``config.tenant``; both
    empty means no ``tenant`` key is injected.
    """
    tenant = override if override is not None else config.tenant
    if tenant:
        kwargs["tenant"] = tenant
    return kwargs


@asynccontextmanager
async def open_session(config: A2AConfig) -> AsyncIterator[tuple[Client, ClientCallContext | None]]:
    """Open a short-lived A2A SDK client for one-shot RPCs.

    Combines the httpx client, card resolution, and SDK factory into a
    single ``async with`` block. The httpx client is closed on exit so
    callers don't have to track it.

    Yields the call context alongside the client. Activation is
    per-request, so every RPC inside the block must pass it as
    ``context=`` or a server requiring an extension rejects the call.
    """
    httpx_client = make_httpx_client(
        headers=dict(config.headers) if config.headers else None,
        timeout=config.timeout,
        factory=config.httpx_client_factory,
    )
    try:
        card = (
            config.preset_card
            or await A2ACardResolver(httpx_client=httpx_client, base_url=config.card_url).get_agent_card()
        )
        iface, transport = select_interface(card, url=config.card_url, prefer=config.prefer)
        validate_protocol_version(iface, url=config.card_url, transport=transport)
        validate_extension_activation(card, config.extensions, url=config.card_url)
        sdk = make_a2a_client(
            card=card,
            httpx_client=httpx_client,
            streaming=False,
            transport=transport,
            interceptors=tuple(config.interceptors),
            grpc_channel_factory=config.grpc_channel_factory,
        )
        yield sdk, extension_call_context(config.extensions)
    finally:
        await httpx_client.aclose()
