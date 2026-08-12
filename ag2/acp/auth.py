# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Authentication seam for :class:`~ag2.acp.agent.ACPAgent`.

AG2 owns the protocol hook — advertising methods at ``initialize`` and gating
session operations until ``authenticate`` succeeds. The embedding application
owns everything behind it: credentials, user identity, authorization scope,
token storage and revocation.

Local stdio needs none of this: the Client launched the process, so the default
is no provider, no advertised methods, and ``authenticate`` rejected. A remote
transport, or the ACP Registry (which requires registered Agents to advertise
valid authentication methods), is where a provider becomes necessary.

.. warning::

   This seam authenticates a connection. It does not identify a tenant.

   :meth:`AuthProvider.authenticate` returns ``None``, so no principal or claims
   reach the turn. Every authenticated Client drives the same agent, the same
   tools and the same :class:`~ag2.knowledge.KnowledgeStore`, and downstream code
   has no trusted identity to authorize against.

   The ACP ``_meta`` map is the only caller-supplied identity that reaches a
   turn. It is client-controlled: suitable as provenance, not as the basis of an
   authorization decision.

   Serve one tenant per :class:`~ag2.acp.agent.ACPAgent`. This matches the stdio
   deployment, where the Client launches the process. Multi-tenant deployments
   require one process and one agent per tenant.

   Carrying a principal and claims into the turn is deferred until a downstream
   consumer defines the requirement, rather than fixing an authentication
   contract that would be costly to change afterwards.
"""

from typing import Any, Protocol, TypeAlias, runtime_checkable

from acp import schema

# The ``InitializeResponse.auth_methods`` union. The SDK exports no named alias
# for it, so — as with the aliases in :mod:`ag2.acp.types` — mirror it once here.
AuthMethod: TypeAlias = schema.EnvVarAuthMethod | schema.TerminalAuthMethod | schema.AuthMethodAgent

__all__ = (
    "AuthMethod",
    "AuthProvider",
    "AuthenticationFailedError",
    "StaticTokenAuth",
)


class AuthenticationFailedError(Exception):
    """Raised by an :class:`AuthProvider` when a credential is rejected."""


@runtime_checkable
class AuthProvider(Protocol):
    """Advertises authentication methods and validates them.

    Implement this to plug an application's identity system into ACP. The
    provider makes the admission decision only; any identity it resolves stays
    within the provider. See the single-tenant warning in this module's
    docstring before serving more than one tenant from one agent.
    """

    def methods(self) -> "list[AuthMethod]":
        """The methods advertised in the ``initialize`` response."""
        ...

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        """Validate an ``authenticate`` call.

        Return normally to accept. Raise :class:`AuthenticationFailedError` (or any
        exception) to reject; the adapter surfaces it as a protocol error.

        The ``None`` return type is intentional: this is an admission gate, not
        an identity lookup. No value from here is carried into the turn, so a
        tool cannot determine which Client authenticated.
        """
        ...


class StaticTokenAuth:
    """Minimal :class:`AuthProvider` checking a shared secret.

    Intended for tests and single-tenant deployments where the Client and Agent
    are configured together. It is *not* a substitute for an identity system:
    there is one credential, it never expires, and it identifies no user.
    """

    __slots__ = ("_token", "_method_id", "_description")

    def __init__(
        self,
        token: str,
        *,
        method_id: str = "token",
        description: str = "Shared token configured on both the Client and the Agent.",
    ) -> None:
        if not token:
            raise ValueError("token must be a non-empty string.")
        self._token = token
        self._method_id = method_id
        self._description = description

    def methods(self) -> "list[AuthMethod]":
        return [
            schema.AuthMethodAgent(
                id=self._method_id,
                name="Shared token",
                description=self._description,
            )
        ]

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        if method_id != self._method_id:
            raise AuthenticationFailedError(f"Unknown authentication method: {method_id!r}")
        supplied = kwargs.get("token") or (kwargs.get("meta") or {}).get("token")
        if supplied != self._token:
            raise AuthenticationFailedError("Invalid token.")
