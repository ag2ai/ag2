# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from a2a.client import ClientCallInterceptor, ClientConfig
from a2a.types import AgentCard
from typing_extensions import Self

from autogen.beta.config.config import ModelConfig

from .cards import url_from_card
from .client import CONTEXT_ID_VAR_KEY, A2AClient
from .types import HttpxClientFactory


@dataclass(slots=True)
class A2AConfig(ModelConfig):
    """``ModelConfig`` connecting an AG2 ``Agent`` to a remote A2A server.

    From the agent loop's point of view, ``Agent(config=A2AConfig(url))`` behaves
    like any other ``LLMClient`` — streaming text and reasoning, ``tool_calls``
    via the optional client-side-tools extension, populated ``Usage``.

    ``client_factory`` ownership: clients returned from the factory remain the
    **caller's** responsibility — ``A2AClient.aclose()`` will not close them.
    Use this when you want to share a single ``httpx.AsyncClient`` across
    multiple ``A2AClient`` instances (e.g. in tests with ``ASGITransport``).
    """

    url: str
    client_factory: HttpxClientFactory | None = None
    client_config: ClientConfig | None = None
    interceptors: tuple[ClientCallInterceptor, ...] = field(default_factory=tuple)
    max_reconnects: int = 3
    reconnect_backoff: float = 0.5
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
            reconnect_backoff=self.reconnect_backoff,
            agent_card=self.agent_card,
        )

    def seed_subtask_variables(self, parent_vars: dict[str, Any]) -> None:
        """Implements :class:`SubtaskContextSeeder`.

        Ensures every sub-task spawned from a parent agent backed by this
        config shares the same A2A ``context_id`` — so the server sees parallel
        and serial calls as one conversation.
        """
        parent_vars.setdefault(CONTEXT_ID_VAR_KEY, uuid4().hex)

    @classmethod
    def from_card(cls, card: AgentCard, /, **overrides: Any) -> Self:
        """Build an ``A2AConfig`` from a pre-fetched ``AgentCard``.

        Saves the ``/.well-known/agent-card.json`` round-trip when the card has
        already been obtained from a discovery registry or static catalog.
        """
        url = overrides.pop("url") if "url" in overrides else url_from_card(card)
        return cls(url=url, agent_card=card, **overrides)
