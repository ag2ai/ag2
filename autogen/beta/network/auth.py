# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Authentication adapters.

V1 ships ``NoAuth`` only — every claim is accepted. ``ApiKeyAuth``
lands in Phase 3 alongside the WebSocket transport. The ``AuthAdapter``
Protocol stays open so further schemes ship additively (JWT, mTLS, and
signed-challenge are AG2 Cloud features).
"""

from typing import Any, Protocol, runtime_checkable

from .errors import AuthError
from .identity import Passport

__all__ = (
    "AuthAdapter",
    "AuthRegistry",
    "NoAuth",
    "default_registry",
)


@runtime_checkable
class AuthAdapter(Protocol):
    """Validates a passport's auth claim at the connection handshake."""

    scheme: str

    async def validate(self, passport: Passport, claim: dict[str, Any]) -> None:
        """Raise ``AuthError`` on failure; return ``None`` on success."""
        ...


class NoAuth:
    """No-op adapter — accepts every claim. V1 default."""

    scheme = "none"

    async def validate(self, passport: Passport, claim: dict[str, Any]) -> None:
        return None


class AuthRegistry:
    """Registry mapping ``scheme`` strings to ``AuthAdapter`` impls."""

    def __init__(self, adapters: list[AuthAdapter]) -> None:
        # __init__ stores params; no side effects.
        self._adapters: dict[str, AuthAdapter] = {a.scheme: a for a in adapters}

    def get(self, scheme: str) -> AuthAdapter:
        try:
            return self._adapters[scheme]
        except KeyError as exc:
            raise AuthError(f"unknown auth scheme: {scheme!r}") from exc

    def schemes(self) -> list[str]:
        return list(self._adapters.keys())


# Default registry — ``NoAuth`` only. Apps wanting ``ApiKeyAuth`` (Phase
# 3) construct their own ``AuthRegistry([NoAuth(), ApiKeyAuth()])`` and
# pass it to ``Hub(... auth=...)``.
default_registry = AuthRegistry([NoAuth()])
