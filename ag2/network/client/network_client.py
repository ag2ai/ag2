# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``NetworkClient`` Protocol — abstract participant in a network.

``AgentClient`` is the built-in implementation (backed by an ``Agent``).
Other participant kinds (e.g. a queue + UI bridge, or operational tools
with no LLM) plug into the same Protocol without inheriting from
``AgentClient`` — implementing the four members below is enough.
"""

from typing import Protocol

from ..envelope import Envelope
from ..identity import Passport, Resume

__all__ = ("NetworkClient",)


class NetworkClient(Protocol):
    """A participant in a network.

    The Protocol surface covers identity, inbound delivery, and
    disconnect. Channel-opening lives on ``AgentClient`` rather than
    the Protocol so non-agent participants (UI bridges, admin tools)
    can refuse to initiate channels if that doesn't make sense for
    them.
    """

    @property
    def agent_id(self) -> str: ...

    @property
    def passport(self) -> Passport: ...

    @property
    def resume(self) -> Resume: ...

    async def receive(self, envelope: Envelope) -> None:
        """Hub delivers an envelope to this participant.

        Implementations translate it into the local execution model —
        ``Agent.ask`` for ``AgentClient``, queue push for
        ``HumanClient``, etc.
        """
        ...

    async def disconnect(self) -> None:
        """Tear down resources. Idempotent."""
        ...
