# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``NetworkClient`` Protocol — abstract participant in a network.

V1 ships ``AgentClient`` as the only implementation (backed by an
``Agent``). Future ``HumanClient`` (queue + UI bridge) and
``AdminClient`` (operational tools, no LLM) plug into the same Protocol
without inheriting from ``AgentClient`` — implementing the four members
below is enough.

Session opening is M2: ``Session`` doesn't exist in the framework yet,
so ``open()`` is intentionally absent from the M1 Protocol surface and
will be added when the session adapter machinery lands.
"""

from typing import Protocol, runtime_checkable

from ..envelope import Envelope
from ..identity import Passport, Resume

__all__ = ("NetworkClient",)


@runtime_checkable
class NetworkClient(Protocol):
    """A participant in a network.

    ``AgentClient`` is the V1 implementation backed by an ``Agent``.
    Future ``HumanClient`` / ``AdminClient`` implement the same
    Protocol. M2 will add ``open(...)`` for session creation; M1 keeps
    the surface to identity + receive + disconnect.
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
