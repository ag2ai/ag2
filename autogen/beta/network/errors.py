# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network-layer exception hierarchy.

Every network error derives from :class:`NetworkError`. Sub-hierarchies are
grouped by where the error originates — handshake / registry / session /
rule / transport / limit / inbox — so callers can handle families uniformly.
"""

from __future__ import annotations


class NetworkError(Exception):
    """Base class for all V3 network errors."""


class RegistryError(NetworkError):
    """Registration or discovery failed."""


class UnknownActorError(RegistryError):
    """Looked up an actor that is not registered with the hub."""


class DuplicateRegistrationError(RegistryError):
    """An actor name is already registered on the hub."""


class AuthError(NetworkError):
    """Identity failed authentication at the hub handshake."""


class RuleViolationError(NetworkError):
    """A send or open was rejected by a rule check."""


class AccessDeniedError(RuleViolationError):
    """Access block triggered (inbound_from / outbound_to / session_types)."""


class LimitExceededError(RuleViolationError):
    """A per-actor limit (concurrent sessions, rate, tokens, cost) was hit."""


class SessionError(NetworkError):
    """Base class for session-lifecycle errors."""


class UnknownSessionError(SessionError):
    """Looked up a session that does not exist or has been archived."""


class SessionClosedError(SessionError):
    """Attempted to operate on a session whose state forbids it."""


class SessionTypeError(SessionError):
    """The session type's delivery rules were violated."""


class InviteRejectedError(SessionError):
    """A participant rejected the invite during handshake."""


class TransportError(NetworkError):
    """Base class for link / transport errors."""


class LinkClosedError(TransportError):
    """The underlying link is closed or disconnected."""


class FrameError(TransportError):
    """A malformed or unexpected frame was seen on the wire."""


class InboxError(NetworkError):
    """Base class for inbox errors."""


class InboxFullError(InboxError):
    """The recipient's inbox is at its configured capacity."""


class TimeoutError(NetworkError):  # noqa: A001 — intentionally shadows builtin in network scope.
    """A ``Session.ask`` or subscription wait exceeded its deadline."""
