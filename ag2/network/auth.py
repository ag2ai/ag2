# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Authentication adapters.

Ships ``NoAuth`` (every claim accepted; the default) and ``ApiKeyAuth``
(constant-time token compare against a static map or async resolver).
The ``AuthAdapter`` Protocol stays open so callers can plug in
alternate schemes (JWT, OAuth2, mTLS, signed-challenge) by passing a
custom ``AuthRegistry``.
"""

import hmac
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, ClassVar, Protocol

from .errors import AuthError
from .identity import Passport

__all__ = (
    "ApiKeyAuth",
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


class ApiKeyAuth:
    """API-key adapter — constant-time token compare.

    Expects the claim to carry ``{"token": "<secret>"}``. The expected
    token is resolved by ``passport.name``:

    * ``keys`` — static ``Mapping[name -> token]`` for in-config keys.
    * ``resolver`` — optional async callable for dynamic lookups
      (database, secret manager, …). Returns ``None`` to signal an
      unknown name. ``keys`` is consulted first; ``resolver`` is the
      fallback.

    Constructed with no entries means every name is unknown, every
    validation fails — useful as a strict-mode default for tests.
    """

    scheme = "api_key"

    def __init__(
        self,
        keys: Mapping[str, str] | None = None,
        *,
        resolver: Callable[[str], Awaitable[str | None]] | None = None,
    ) -> None:
        # __init__ stores params; no side effects.
        self._keys: dict[str, str] = dict(keys) if keys is not None else {}
        self._resolver = resolver

    async def validate(self, passport: Passport, claim: dict[str, Any]) -> None:
        token = claim.get("token")
        if not isinstance(token, str) or not token:
            raise AuthError("api_key claim missing required string field 'token'")

        expected = self._keys.get(passport.name)
        if expected is None and self._resolver is not None:
            expected = await self._resolver(passport.name)
        if expected is None:
            raise AuthError(f"no api_key registered for {passport.name!r}")

        # Constant-time compare so token rejection latency does not leak
        # which prefix matched. ``compare_digest`` requires equal-length
        # bytes; encode both sides as UTF-8.
        if not hmac.compare_digest(expected.encode("utf-8"), token.encode("utf-8")):
            raise AuthError(f"api_key mismatch for {passport.name!r}")


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
