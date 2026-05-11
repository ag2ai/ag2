# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Authentication adapters.

Ships ``NoAuth`` only — every claim is accepted. The ``AuthAdapter``
Protocol stays open so callers can plug in alternate schemes (API key,
JWT, mTLS, signed-challenge) by passing a custom ``AuthRegistry``.
"""

from typing import Any, ClassVar, Protocol

from .errors import AuthError
from .identity import Passport

__all__ = (
    "AuthAdapter",
    "AuthRegistry",
    "NoAuth",
)


class AuthAdapter(Protocol):
    """Validates a passport's auth claim at the connection handshake."""

    scheme: str

    async def validate(self, passport: Passport, claim: dict[str, Any]) -> None:
        """Raise ``AuthError`` on failure; return ``None`` on success."""
        ...


class NoAuth:
    """No-op adapter — accepts every claim. Default registry entry."""

    scheme = "none"

    async def validate(self, passport: Passport, claim: dict[str, Any]) -> None:
        return None


class AuthRegistry:
    """Registry mapping ``scheme`` strings to ``AuthAdapter`` impls.

    Apps wanting a custom adapter construct their own
    ``AuthRegistry([NoAuth(), MyAuth()])`` and pass it to
    ``Hub(... auth=...)``. Use :meth:`default` for the ``NoAuth``-only
    default.
    """

    _DEFAULT: ClassVar["AuthRegistry | None"] = None

    def __init__(self, adapters: list[AuthAdapter]) -> None:
        # __init__ stores params; no side effects.
        self._adapters: dict[str, AuthAdapter] = {a.scheme: a for a in adapters}

    @classmethod
    def default(cls) -> "AuthRegistry":
        """Return the lazily-initialised default registry — ``NoAuth`` only."""
        if cls._DEFAULT is None:
            cls._DEFAULT = cls([NoAuth()])
        return cls._DEFAULT

    def get(self, scheme: str) -> AuthAdapter:
        try:
            return self._adapters[scheme]
        except KeyError as exc:
            raise AuthError(f"unknown auth scheme: {scheme!r}") from exc

    def schemes(self) -> list[str]:
        return list(self._adapters.keys())
