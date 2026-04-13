# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AuthAdapter protocol + the Phase 3a adapter zoo.

An auth adapter is a pluggable validator selected at handshake by
``ActorIdentity.auth.scheme``. A hub can install any mix of adapters
via :class:`AuthRegistry`; each adapter registers itself under a
stable scheme name and the hub picks the matching one per-identity.

Phase 1 shipped only :class:`NoAuth`. Phase 3a adds two real
validators — :class:`ApiKeyAuth` (fingerprint allowlist, the
dev-friendly default for hosted deployments) and :class:`JwtAuth`
(HMAC/RSA JWS verification against a shared secret or public key).
Both validate at the WS ``hello`` frame and at the HTTP front door,
in both cases before rule evaluation.

Future adapters (``MtlsAuth``, ``SignedChallengeAuth``) plug into
the same registry without protocol changes.
"""

from __future__ import annotations

import hmac
import logging
from typing import Any, Protocol, runtime_checkable

from .errors import AuthError
from .identity import ActorIdentity

log = logging.getLogger("autogen.beta.network.auth")


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


# ---------------------------------------------------------------------------
# ApiKeyAuth — fingerprint allowlist
# ---------------------------------------------------------------------------


class ApiKeyAuth:
    """Per-hub fingerprint allowlist.

    Each identity's ``auth.key_fingerprint`` is a hex digest (e.g.
    ``sha256:...``) the operator has registered with the hub. The
    matching ``claim`` must carry a ``fingerprint`` field equal to
    the identity's fingerprint AND the fingerprint must be in the
    hub's installed allowlist.

    This is a dev-friendly scheme that does not require asymmetric
    keys or a full OIDC dance — the operator manages a small set of
    shared secrets (one per actor or one per deployment) and
    distributes them out-of-band. ``fingerprint`` equality uses
    :func:`hmac.compare_digest` so timing attacks on the comparison
    are not feasible.

    Example::

        adapter = ApiKeyAuth(
            allowed_fingerprints={
                "sha256:ab12...": "alice",
                "sha256:cd34...": "bob",
            }
        )
        hub = Hub(store, auth=AuthRegistry([adapter]))

    A None / empty allowlist accepts any fingerprint that matches
    the identity declaration — useful for tests that want the full
    ``api_key`` scheme path without the admin burden.
    """

    scheme: str = "api_key"

    def __init__(
        self,
        allowed_fingerprints: dict[str, str] | None = None,
    ) -> None:
        self._allowlist: dict[str, str] = dict(allowed_fingerprints or {})

    def add_fingerprint(self, fingerprint: str, owner: str) -> None:
        """Add a fingerprint → owner mapping to the allowlist."""

        self._allowlist[fingerprint] = owner

    def revoke_fingerprint(self, fingerprint: str) -> None:
        self._allowlist.pop(fingerprint, None)

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        if identity.auth.scheme != self.scheme:
            raise AuthError(
                f"ApiKeyAuth cannot validate scheme {identity.auth.scheme!r}"
            )

        declared_fp = identity.auth.key_fingerprint
        if not declared_fp:
            raise AuthError("ApiKeyAuth: identity.auth.key_fingerprint is required")

        presented_fp = claim.get("fingerprint")
        if not isinstance(presented_fp, str):
            raise AuthError("ApiKeyAuth: auth_claim must carry 'fingerprint' string")

        # Constant-time compare so a malicious probe can't infer the
        # fingerprint byte-by-byte from response timing.
        if not hmac.compare_digest(declared_fp, presented_fp):
            raise AuthError("ApiKeyAuth: presented fingerprint does not match identity")

        # If the hub has an allowlist, enforce it. An empty allowlist
        # means "any correctly-matching fingerprint is OK" — useful for
        # tests; production hubs should always populate the allowlist.
        if self._allowlist and declared_fp not in self._allowlist:
            raise AuthError("ApiKeyAuth: fingerprint is not on the hub's allowlist")


# ---------------------------------------------------------------------------
# JwtAuth — JWS verification
# ---------------------------------------------------------------------------


class JwtAuth:
    """JWT/JWS verification against a shared secret or public key.

    The identity's ``auth.issuer`` and ``auth.audience`` fields are
    the expected ``iss`` and ``aud`` claims in the token. The
    ``claim`` dict must carry a ``token`` string (the encoded JWS).

    Constructor parameters:

    * ``key`` — the verification key. For ``HS*`` algorithms this is
      a shared secret ``str`` or ``bytes``; for ``RS*`` / ``ES*`` it
      is a PEM-encoded public key.
    * ``algorithms`` — the list of allowed algorithms. Default is
      ``["HS256"]`` to keep the first-install experience simple;
      production deployments should override with the right list
      for their signing infrastructure.
    * ``required_issuer`` / ``required_audience`` — defaults pulled
      from the identity at validation time but can be overridden
      per-adapter instance to force a single-tenant policy.
    * ``leeway`` — seconds of clock skew allowed on ``exp`` / ``nbf``
      / ``iat``. Default 60.

    The adapter raises :class:`AuthError` with a stable message
    family so hub logs and HTTP 401 responses can categorize
    failures without leaking token details.
    """

    scheme: str = "jwt"

    def __init__(
        self,
        *,
        key: str | bytes,
        algorithms: list[str] | None = None,
        required_issuer: str | None = None,
        required_audience: str | None = None,
        leeway: int = 60,
    ) -> None:
        self._key = key
        self._algorithms = list(algorithms) if algorithms else ["HS256"]
        self._issuer = required_issuer
        self._audience = required_audience
        self._leeway = leeway

    async def validate(self, identity: ActorIdentity, claim: dict[str, Any]) -> None:
        if identity.auth.scheme != self.scheme:
            raise AuthError(f"JwtAuth cannot validate scheme {identity.auth.scheme!r}")

        token = claim.get("token")
        if not isinstance(token, str) or not token:
            raise AuthError("JwtAuth: auth_claim must carry 'token' string")

        try:
            import jwt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise AuthError(
                "JwtAuth requires the 'PyJWT' library. "
                "Install with: pip install PyJWT"
            ) from exc

        expected_issuer = self._issuer or identity.auth.issuer
        expected_audience = self._audience or identity.auth.audience

        try:
            payload = jwt.decode(
                token,
                self._key,
                algorithms=self._algorithms,
                audience=expected_audience,
                issuer=expected_issuer,
                leeway=self._leeway,
                options={
                    "require": ["exp", "iat"] if expected_audience else ["exp"],
                },
            )
        except Exception as exc:
            raise AuthError(f"JwtAuth: token verification failed: {exc}") from exc

        # Minimal well-formedness: the token's subject should match
        # the identity's name so a stolen token for one actor can't
        # be replayed against another registration.
        sub = payload.get("sub")
        if sub and sub != identity.name:
            raise AuthError(
                f"JwtAuth: token 'sub' ({sub!r}) does not match identity name "
                f"({identity.name!r})"
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


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
    Production deployments install :class:`ApiKeyAuth` or
    :class:`JwtAuth` (or both) via ``Hub(auth=AuthRegistry([...]))``.
    """

    return AuthRegistry(adapters=[NoAuth()])


# Convenience: a "mixed" registry with every Phase 3a adapter
# pre-installed. Mostly for tests — production hubs pick the specific
# adapters they want.
def dev_registry(
    *,
    api_key_allowlist: dict[str, str] | None = None,
    jwt_key: str | bytes | None = None,
    jwt_algorithms: list[str] | None = None,
) -> AuthRegistry:
    """Convenience factory that installs NoAuth + ApiKeyAuth + optional JwtAuth."""

    adapters: list[AuthAdapter] = [
        NoAuth(),
        ApiKeyAuth(allowed_fingerprints=api_key_allowlist),
    ]
    if jwt_key is not None:
        adapters.append(JwtAuth(key=jwt_key, algorithms=jwt_algorithms))
    return AuthRegistry(adapters=adapters)
