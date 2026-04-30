# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from a2a.client import ClientCallInterceptor, ClientConfig
from a2a.types import AgentCard
from typing_extensions import Self

from autogen.beta.config.config import ModelConfig

from .a2a_client import CONTEXT_ID_VAR_KEY, A2AClient
from .types import HttpxClientFactory


@dataclass(slots=True)
class A2AConfig(ModelConfig):
    url: str
    client_factory: HttpxClientFactory | None = None
    client_config: ClientConfig | None = None
    interceptors: tuple[ClientCallInterceptor, ...] = field(default_factory=tuple)
    max_reconnects: int = 3
    reconnect_backoff: float = 0.5
    agent_card: AgentCard | None = None

    # Variables keys that subagent tools should pre-seed in the parent context
    # so that all tool calls to this agent share one A2A `context_id` (per the
    # protocol's "context_id maintains context across related tasks" rule).
    # Read by `subagent_tool` via duck-typing — no hard dep from tools/ on a2a/.
    _subtask_propagate_keys: ClassVar[tuple[str, ...]] = (CONTEXT_ID_VAR_KEY,)

    def copy(self, /, **overrides: Any) -> Self:
        return replace(self, **overrides)

    def create(self) -> A2AClient:
        return A2AClient(
            url=self.url,
            client_factory=self.client_factory,
            client_config=self.client_config,
            interceptors=list(self.interceptors),
            max_reconnects=self.max_reconnects,
            reconnect_backoff=self.reconnect_backoff,
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
