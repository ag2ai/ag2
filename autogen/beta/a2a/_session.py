# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from a2a.client import Client

from .config import A2AConfig
from .transports._http import fetch_card, make_a2a_client, make_httpx_client


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
async def open_session(config: A2AConfig) -> AsyncIterator[Client]:
    """Open a short-lived A2A SDK client for one-shot RPCs.

    Combines the httpx client, card resolution, and SDK factory into a
    single ``async with`` block. The httpx client is closed on exit so
    callers don't have to track it.
    """
    httpx_client = make_httpx_client(
        headers=dict(config.headers) if config.headers else None,
        timeout=config.timeout,
        factory=config.httpx_client_factory,
    )
    try:
        card = config.preset_card or await fetch_card(httpx_client, url=config.url)
        sdk = make_a2a_client(
            card=card,
            httpx_client=httpx_client,
            streaming=False,
            transports=tuple(config.transports),
            interceptors=tuple(config.interceptors),
            grpc_channel_factory=config.grpc_channel_factory,
        )
        yield sdk
    finally:
        await httpx_client.aclose()
