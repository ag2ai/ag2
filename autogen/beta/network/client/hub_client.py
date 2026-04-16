# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""HubClient — outbound handle for registration, discovery, and session creation.

Phase 1 holds a direct reference to the :class:`Hub` (the control-plane
code path) plus a :class:`LocalLink` (the stateful code path). Phase 3
swaps the hub ref for an HTTP client and the link for :class:`WsLink`
without touching ``HubClient`` consumers.
"""

from __future__ import annotations

from typing import Any

from ..hub import Hub
from ..identity import ActorIdentity
from ..rule import Rule
from ..transport.link import Link
from .actor_client import ActorClient
from .human import HumanClient, HumanSurface


class HubClient:
    """Connection to a hub. Produces :class:`ActorClient` per registration."""

    def __init__(self, hub: Hub, link: Link) -> None:
        self._hub = hub
        self._link = link
        self._actor_clients: list[ActorClient] = []

    # ------------------------------------------------------------------

    async def register(
        self,
        actor: Any,
        *,
        identity: ActorIdentity,
        rule: Rule | None = None,
        auth_claim: dict[str, Any] | None = None,
        install_stdlib_transforms: bool = True,
    ) -> ActorClient:
        """Register ``actor`` under ``identity`` and return an ``ActorClient``.

        ``install_stdlib_transforms`` controls whether the Phase 5a.1
        named-transform standard library (``redact_pii`` /
        ``truncate_long_content`` / ``stamp_audit_header``) is
        pre-registered on the new client's :class:`TransformRegistry`.
        Default True — opt out for tests that need a pristine registry
        or for deployments that want to install a custom stdlib first.
        """

        stamped = await self._hub.register(identity, rule, auth_claim=auth_claim)
        client = ActorClient(
            actor=actor,
            identity=stamped,
            rule=rule if rule is not None else await self._hub.get_rule(stamped.actor_id or ""),
            hub=self._hub,
            link=self._link,
            hub_client=self,
            install_stdlib_transforms=install_stdlib_transforms,
        )
        await client._start()
        self._actor_clients.append(client)
        return client

    async def register_human(
        self,
        *,
        surface: HumanSurface,
        identity: ActorIdentity,
        rule: Rule | None = None,
        auth_claim: dict[str, Any] | None = None,
        actor: Any = None,
    ) -> HumanClient:
        """Register a human identity with the hub and return a :class:`HumanClient`.

        Mirrors :meth:`register` but produces a :class:`HumanClient`
        wired to ``surface`` instead of a regular
        :class:`ActorClient`. The identity's ``runtime_kind`` is
        automatically set to ``"human"`` if it is still at the
        default ``"python"`` — explicit overrides are preserved so
        callers can ship ``"browser"`` or ``"external"`` identities
        that happen to use the same human surface protocol.

        Design §13.5: the hub has no idea this registration is a
        human. From its perspective it is just another identity
        with its own rule, its own inbox, and its own transport.
        The only specialization happens in this Python process.
        """

        if identity.runtime_kind == "python":
            identity.runtime_kind = "human"

        stamped = await self._hub.register(identity, rule, auth_claim=auth_claim)
        applied_rule = (
            rule
            if rule is not None
            else await self._hub.get_rule(stamped.actor_id or "")
        )
        client = HumanClient(
            surface=surface,
            identity=stamped,
            rule=applied_rule,
            hub=self._hub,
            link=self._link,
            hub_client=self,
            actor=actor,
        )
        await client._start()
        self._actor_clients.append(client)
        return client

    async def find(
        self,
        capability: str | None = None,
        *,
        query: str | None = None,
    ) -> list[ActorIdentity]:
        return await self._hub.find(capability=capability, query=query)

    async def describe(self, name_or_id: str) -> ActorIdentity:
        return await self._hub.describe(name_or_id)

    async def close(self) -> None:
        """Disconnect every ActorClient without unregistering them.

        Registrations live on the hub and can be re-attached on a
        subsequent :meth:`register` call. Use :meth:`shutdown` when you
        want to tear down identity state for good.
        """

        for client in list(self._actor_clients):
            await client.disconnect()
        self._actor_clients.clear()

    async def shutdown(self) -> None:
        """Unregister every ActorClient and release all hub state."""

        for client in list(self._actor_clients):
            await client.unregister()
        self._actor_clients.clear()
