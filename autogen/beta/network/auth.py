# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AuthAdapter protocol + Phase 1 ``NoAuth`` implementation.

An auth adapter is a pluggable validator selected at handshake by
``ActorIdentity.auth.scheme``. Phase 1 ships only ``NoAuth`` — the plugin
point exists so Phase 3 can add ``ApiKeyAuth`` / ``JwtAuth`` / ``MtlsAuth``
without a protocol change.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .errors import AuthError
from .identity import ActorIdentity


@runtime_checkable
class AuthAdapter(Protocol):
    """Validates an incoming :class:`ActorIdentity` at hub handshake."""

    scheme: str

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        """Raise :class:`AuthError` if the identity cannot be authenticated.

        ``claim`` carries scheme-specific runtime data (bearer token, mTLS
        peer cert fingerprint, signed-challenge response). The adapter is
        responsible for comparing ``claim`` against ``identity.auth.claim``
        according to its scheme's rules.
        """
        ...


class NoAuth:
    """Pass-through adapter. Accepts any identity whose scheme is ``none``."""

    scheme: str = "none"

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        # Phase 1: no auth enforcement. We still check that the identity
        # actually declared ``none`` so a mis-routed identity (e.g. one
        # expecting ``jwt`` that landed on a NoAuth-only hub) fails loudly.
        if identity.auth.scheme not in ("none", ""):
            raise AuthError(
                f"NoAuth adapter cannot validate scheme {identity.auth.scheme!r} "
                "— install the matching adapter on the hub."
            )


class AuthRegistry:
    """Maps an identity's ``auth.scheme`` to the installed adapter.

    Phase 1 lets a hub register multiple adapters; missing-scheme lookups
    raise :class:`AuthError` so hubs can refuse unknown schemes rather than
    silently fall through to no-auth.
    """

    def __init__(self, adapters: list[AuthAdapter] | None = None) -> None:
        self._adapters: dict[str, AuthAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: AuthAdapter) -> None:
        self._adapters[adapter.scheme] = adapter

    def get(self, scheme: str) -> AuthAdapter:
        try:
            return self._adapters[scheme]
        except KeyError as exc:
            raise AuthError(f"No auth adapter installed for scheme {scheme!r}") from exc

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        scheme = identity.auth.scheme or "none"
        adapter = self.get(scheme)
        await adapter.validate(identity, claim)

    def schemes(self) -> list[str]:
        return sorted(self._adapters)


def default_registry() -> AuthRegistry:
    """Default registry used by ``Hub`` when none is supplied.

    Ships ``NoAuth`` so OSS hubs run locally with zero configuration.
    """

    return AuthRegistry(adapters=[NoAuth()])
