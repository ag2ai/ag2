# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``RemoteAgentProxy`` Protocol — federation dispatch seam.

A :class:`RemoteAgentProxy` carries envelopes from the local hub to a
participant whose passport has ``kind="remote_agent"``. The hub keeps
a registry of proxies keyed by :attr:`scheme` (typically matching the
remote participant's ``AuthBlock.scheme`` — ``"a2a"``, ``"grpc"``,
``"kafka"``, …). When ``_dispatch`` encounters a remote recipient, it
looks up the proxy by scheme and delegates instead of sending a
``NotifyFrame`` to a local endpoint.

No proxies ship in the framework — they are tenant impls.
``examples/network_a2a_bridge/`` (separate followup) carries a worked
A2A implementation as proof of extensibility.
"""

from typing import Protocol, runtime_checkable

from .envelope import Envelope
from .identity import Passport

__all__ = ("RemoteAgentProxy",)


@runtime_checkable
class RemoteAgentProxy(Protocol):
    """Dispatches envelopes from the local hub to a remote participant.

    Implementations encode :class:`Envelope` into the target wire
    protocol (A2A, gRPC, raw HTTP, message bus, …), perform any
    transport-side retries, and return when the envelope has been
    handed off. Any exception raised propagates back to the hub and
    is surfaced through :meth:`HubListener.on_dispatch_failed`; the
    proxy does not need to fan out observability itself.

    ``scheme`` is the registry key. The hub looks up proxies via the
    recipient passport's ``auth.scheme``, so a remote agent whose
    passport carries ``AuthBlock(scheme="a2a", ...)`` routes through
    the proxy registered with ``scheme = "a2a"``.

    ``close()`` is called by :meth:`Hub.close` (and may be called
    explicitly by tenants) to release any persistent transport
    resources — connection pools, background tasks, etc.
    """

    scheme: str

    async def dispatch(self, envelope: Envelope, recipient: Passport) -> None:
        """Deliver ``envelope`` to ``recipient`` over the proxy's transport."""
        ...

    async def close(self) -> None:
        """Release transport resources held by the proxy. Idempotent."""
        ...
