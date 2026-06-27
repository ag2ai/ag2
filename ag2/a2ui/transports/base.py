# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""The :class:`A2UITransport` protocol: a deployment's single wire encoding.

A transport owns its HTTP route(s), parses the incoming envelope, runs one turn
through the shared :class:`~ag2.a2ui.dispatch._A2UITurnCore`, and
encodes the outgoing frames in its own wire format. One :class:`A2UIServer` ==
one transport; mixing transports in one process is an anti-pattern — run N
instances behind a reverse-proxy instead.
"""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from starlette.routing import Route

    from ..dispatch import _A2UITurnCore


@runtime_checkable
class A2UITransport(Protocol):
    """A single A2UI wire encoding, plugged into :class:`A2UIServer` via ``transport=``."""

    def routes(self, core: "_A2UITurnCore") -> "list[Route]":
        """Return the Starlette route(s) serving this transport, bound to ``core``.

        ``core`` is the shared turn engine (agent + runtime + clickable actions);
        the transport calls ``core.run_turn(request)`` to drive one turn and
        encodes the yielded frames in its own format.
        """
        ...


__all__ = ("A2UITransport",)
