# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TypedDict

import httpx
from typing_extensions import Self, Unpack

from ag2.config.config import ModelConfig

from .client import NlipClient


class NlipConfigOverrides(TypedDict, total=False):
    url: str
    headers: Mapping[str, str] | None
    timeout: float | None
    max_retries: int
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None


@dataclass(slots=True)
class NlipConfig(ModelConfig):
    """Connection config for a remote NLIP agent acting as an LLM provider.

    ``url`` is the base URL of the NLIP server (e.g. ``"http://host:8000"``);
    requests are posted to ``{url}/nlip/``. NLIP has no discovery card and no
    task lifecycle — every call is a single stateless request/response
    exchange.

    ``httpx_client_factory`` builds the ``httpx.AsyncClient`` used for each
    request; defaults to a plain client constructed from ``headers`` /
    ``timeout``. Override it (e.g. with ``httpx.ASGITransport``) for
    in-process testing — see :mod:`ag2.nlip.testing`.
    """

    url: str
    headers: Mapping[str, str] | None = None
    timeout: float | None = 60.0
    max_retries: int = 3
    httpx_client_factory: Callable[[], httpx.AsyncClient] | None = field(default=None, repr=False)

    def copy(self, /, **overrides: Unpack[NlipConfigOverrides]) -> Self:
        return replace(self, **overrides)

    def create(self) -> NlipClient:
        return NlipClient(
            url=self.url,
            headers=self.headers,
            timeout=self.timeout,
            max_retries=self.max_retries,
            httpx_client_factory=self.httpx_client_factory,
        )


__all__: tuple[str, ...] = ("NlipConfig",)
