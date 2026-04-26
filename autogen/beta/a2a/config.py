# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from typing import Any

from a2a.client import ClientCallInterceptor, ClientConfig
from a2a.types import AgentCard
from typing_extensions import Self

from autogen.beta.config.config import ModelConfig

from .a2a_client import A2AClient, HttpxClientFactory


@dataclass(slots=True)
class A2AConfig(ModelConfig):
    url: str
    client_factory: HttpxClientFactory | None = None
    client_config: ClientConfig | None = None
    interceptors: tuple[ClientCallInterceptor, ...] = field(default_factory=tuple)
    max_reconnects: int = 3
    polling_interval: float = 0.5
    agent_card: AgentCard | None = None

    def copy(self, /, **overrides: Any) -> Self:
        return replace(self, **overrides)

    def create(self) -> A2AClient:
        return A2AClient(
            url=self.url,
            client_factory=self.client_factory,
            client_config=self.client_config,
            interceptors=list(self.interceptors),
            max_reconnects=self.max_reconnects,
            polling_interval=self.polling_interval,
            agent_card=self.agent_card,
        )

    @classmethod
    def from_card(cls, card: AgentCard, /, **overrides: Any) -> Self:
        """Build an `A2AConfig` from a pre-fetched `AgentCard`.

        Useful when the card has already been obtained from a discovery
        registry or loaded manually — saves a round-trip to
        `/.well-known/agent-card.json`.
        """
        url = overrides.pop("url") if "url" in overrides else card.url
        return cls(url=url, agent_card=card, **overrides)
