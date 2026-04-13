# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""ActorIdentity — registration input for an actor.

A single Python ``Actor`` is identity-less. When an actor joins a hub it
presents an :class:`ActorIdentity` that combines who it is (profile), what it
can do (capability surface), and how it authenticates (auth block). The hub
stamps a fresh ``actor_id`` at registration and writes the identity under
``hub/actors/{actor_id}/identity.json``. Two identities from the same Python
actor yield two distinct ``actor_id`` values — each registration is
independent.

Identity is immutable for the life of a registration. Anything that changes
over time (binding, heartbeat, last-known transport) lives in
``runtime.json`` and is owned by the hub — never by the actor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class AuthBlock:
    """How the hub validates this identity at handshake.

    The hub-side :class:`~autogen.beta.network.auth.AuthAdapter` is selected
    by ``scheme`` and receives the full block (including ``claim``) for
    validation.
    """

    scheme: str = "none"
    issuer: str | None = None
    audience: str | None = None
    key_fingerprint: str | None = None
    claim: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "scheme": self.scheme,
            "claim": dict(self.claim),
        }
        if self.issuer is not None:
            data["issuer"] = self.issuer
        if self.audience is not None:
            data["audience"] = self.audience
        if self.key_fingerprint is not None:
            data["key_fingerprint"] = self.key_fingerprint
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthBlock:
        return cls(
            scheme=data.get("scheme", "none"),
            issuer=data.get("issuer"),
            audience=data.get("audience"),
            key_fingerprint=data.get("key_fingerprint"),
            claim=dict(data.get("claim", {})),
        )


@dataclass(slots=True)
class ActorIdentity:
    """Registration input for an actor.

    ``actor_id`` is assigned by the hub at registration and is ``None`` on
    construction. Mutating an identity post-registration is not supported;
    re-register with a new identity instead.
    """

    name: str
    owner: str = ""
    version: str = "1"
    display: str | None = None
    framework: str = "ag2-beta"
    runtime_kind: str = "python"
    model_hint: str | None = None
    locale: str | None = None
    timezone: str | None = None
    capabilities: list[str] = field(default_factory=list)
    summary: str = ""
    domains: list[str] = field(default_factory=list)
    strengths: str = ""
    skill_md: str | None = None
    auth: AuthBlock = field(default_factory=AuthBlock)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Hub-stamped. ``None`` before registration completes.
    actor_id: str | None = None

    def with_actor_id(self, actor_id: str) -> ActorIdentity:
        """Return a copy of this identity with the given ``actor_id`` set."""

        data = asdict(self)
        data["actor_id"] = actor_id
        data["auth"] = AuthBlock(**data["auth"])
        return ActorIdentity(**data)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "owner": self.owner,
            "version": self.version,
            "framework": self.framework,
            "runtime_kind": self.runtime_kind,
            "capabilities": list(self.capabilities),
            "summary": self.summary,
            "domains": list(self.domains),
            "strengths": self.strengths,
            "auth": self.auth.to_dict(),
            "metadata": dict(self.metadata),
        }
        if self.display is not None:
            data["display"] = self.display
        if self.model_hint is not None:
            data["model_hint"] = self.model_hint
        if self.locale is not None:
            data["locale"] = self.locale
        if self.timezone is not None:
            data["timezone"] = self.timezone
        if self.skill_md is not None:
            data["skill_md"] = self.skill_md
        if self.actor_id is not None:
            data["actor_id"] = self.actor_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActorIdentity:
        return cls(
            name=data["name"],
            owner=data.get("owner", ""),
            version=data.get("version", "1"),
            display=data.get("display"),
            framework=data.get("framework", "ag2-beta"),
            runtime_kind=data.get("runtime_kind", "python"),
            model_hint=data.get("model_hint"),
            locale=data.get("locale"),
            timezone=data.get("timezone"),
            capabilities=list(data.get("capabilities", [])),
            summary=data.get("summary", ""),
            domains=list(data.get("domains", [])),
            strengths=data.get("strengths", ""),
            skill_md=data.get("skill_md"),
            auth=AuthBlock.from_dict(data.get("auth", {})),
            metadata=dict(data.get("metadata", {})),
            actor_id=data.get("actor_id"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> ActorIdentity:
        return cls.from_dict(json.loads(payload))
