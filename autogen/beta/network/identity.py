# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identity records: ``Passport``, ``Resume``, ``AgentRuntime``.

Three records back every registered agent — see ``design/identity.md``:

* ``Passport`` — immutable id + billing facts. Hub-stamps ``agent_id``
  at registration.
* ``Resume`` — capability claims and observed track record. Mutates
  over time via tenant-driven ``set_resume`` and hub-driven
  ``record_observation``.
* ``SKILL.md`` — Markdown document with Anthropic frontmatter, stored
  as plain text in M1 (frontmatter parser arrives in M3).

``AgentRuntime`` is hub-owned bookkeeping for the current connection
(transport binding, last heartbeat). It lives next to the identity
files but readers should treat it as cache-only.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = (
    "AgentRuntime",
    "AuthBlock",
    "CostProfile",
    "ObservedStat",
    "Passport",
    "Resume",
    "ResumeExample",
)


@dataclass(slots=True)
class CostProfile:
    """Optional billing / routing hints. None of these fields are validated by V1."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    latency_tier: str | None = None  # "fast" | "balanced" | "deep"


@dataclass(slots=True)
class AuthBlock:
    """How the hub validates this identity at the connection handshake."""

    scheme: str = "none"  # "none" | "api_key" (Phase 3) | future
    issuer: str | None = None
    audience: str | None = None
    key_fingerprint: str | None = None
    claim: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Passport:
    """Immutable identity + billing record for one registration.

    ``agent_id`` is hub-stamped at registration. Mutating any field
    requires unregister + re-register, which yields a fresh ``agent_id``.
    """

    name: str  # human/LLM-facing address; unique per hub
    owner: str = ""
    provider: str | None = None  # "anthropic" | "openai" | None
    model: str | None = None
    cost: CostProfile | None = None
    region: str | None = None
    auth: AuthBlock = field(default_factory=AuthBlock)
    version: int = 1

    # Hub-stamped at registration. None on construction.
    agent_id: str | None = None
    created_at: str = ""  # ISO-Z, hub-stamped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Passport":
        payload = dict(data)
        if isinstance(payload.get("auth"), dict):
            payload["auth"] = AuthBlock(**payload["auth"])
        if isinstance(payload.get("cost"), dict):
            payload["cost"] = CostProfile(**payload["cost"])
        return cls(**payload)


@dataclass(slots=True)
class ResumeExample:
    title: str
    outcome: str = ""  # "completed" | "failed" | free-form
    task_id: str | None = None
    session_id: str | None = None
    when: str | None = None
    note: str = ""


@dataclass(slots=True)
class ObservedStat:
    """Hub-derived per-capability track record. Updated on terminal task events."""

    n: int = 0
    completed: int = 0
    failed: int = 0
    expired: int = 0
    p50_latency_ms: int | None = None


@dataclass(slots=True)
class Resume:
    """Mutable capability claim + observed track record.

    Tenant code provides ``claimed_capabilities``, ``domains``,
    ``summary``, and ``examples`` at registration. The hub mutates
    ``observed`` on terminal task events (M3); tenant code may also
    replace the resume via ``Hub.set_resume(...)``.
    """

    claimed_capabilities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    summary: str = ""  # one-line, indexed for `peers(action="find")`
    examples: list[ResumeExample] = field(default_factory=list)
    observed: dict[str, ObservedStat] = field(default_factory=dict)
    version: int = 1
    last_updated: str = ""  # ISO-Z, hub-stamped on every mutation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resume":
        payload = dict(data)
        if "examples" in payload:
            payload["examples"] = [
                ResumeExample(**e) if isinstance(e, dict) else e for e in payload["examples"]
            ]
        if "observed" in payload:
            payload["observed"] = {
                k: ObservedStat(**v) if isinstance(v, dict) else v
                for k, v in payload["observed"].items()
            }
        return cls(**payload)


@dataclass(slots=True)
class AgentRuntime:
    """Hub-owned per-connection bookkeeping for a registered agent.

    Re-read on hub failover only; rewritten on every heartbeat. Identity
    readers should not depend on its freshness.
    """

    agent_id: str
    binding: str  # "local" | "ws"
    target: str  # opaque endpoint id (local queue id / ws connection id)
    reachable: bool
    last_heartbeat: str  # ISO-Z
