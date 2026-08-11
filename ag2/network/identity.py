# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Identity records: ``Passport``, ``Resume``, ``AgentRuntime``.

Three records back every registered agent:

* ``Passport`` — immutable id + billing facts. Hub-stamps ``agent_id``
  at registration.
* ``Resume`` — capability claims and observed track record. Mutates
  over time via tenant-driven ``set_resume`` and hub-driven
  ``record_observation``.
* ``SKILL.md`` — Markdown document with Anthropic-style frontmatter,
  parsed by ``client/skill_render.py``.

``AgentRuntime`` is hub-owned bookkeeping for the current connection
(transport binding, last heartbeat). It lives next to the identity
files but readers should treat it as cache-only.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

__all__ = (
    "PASSPORT_KINDS",
    "AgentRuntime",
    "AuthBlock",
    "CostProfile",
    "ObservedStat",
    "Passport",
    "PassportKind",
    "Resume",
    "ResumeExample",
)


PassportKind = Literal["agent", "human", "remote_agent"]
PASSPORT_KINDS: tuple[str, ...] = get_args(PassportKind)


@dataclass(slots=True)
class CostProfile:
    """Optional billing / routing hints. None of these fields are validated by V1."""

    input_per_mtok: float | None = None
    output_per_mtok: float | None = None
    latency_tier: str | None = None  # "fast" | "balanced" | "deep"


@dataclass(slots=True)
class AuthBlock:
    """How the hub validates this identity at the connection handshake."""

    scheme: str = "none"  # "none" | future schemes
    issuer: str | None = None
    audience: str | None = None
    key_fingerprint: str | None = None
    claim: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Passport:
    """Immutable identity + billing record for one registration.

    ``agent_id`` is hub-stamped at registration. Mutating any field
    requires unregister + re-register, which yields a fresh ``agent_id``.

    ``kind`` discriminates participant types — ``"agent"`` (default LLM
    participant), ``"human"`` (out-of-band non-LLM participant driven by
    an external UI), or ``"remote_agent"`` (a participant that lives on
    another hub; the local hub holds the passport as a cache and
    dispatches to it via a registered ``RemoteAgentProxy``). ``None``
    is treated as ``"agent"`` for back-compat with passports persisted
    before this field existed.

    ``hub_id`` is ``None`` for every locally-registered participant.
    For remote participants the local hub caches, it carries the
    originating hub's identifier so the canonical URN form is
    ``f"hub://{hub_id}/{agent_id}"``.
    """

    name: str  # human/LLM-facing address; unique per hub
    owner: str = ""
    provider: str | None = None  # "anthropic" | "openai" | None
    model: str | None = None
    cost: CostProfile | None = None
    region: str | None = None
    auth: AuthBlock = field(default_factory=AuthBlock)
    kind: PassportKind | None = None  # None ≡ "agent" for back-compat
    hub_id: str | None = None  # None for local registrations; set for federated peers
    version: int = 1

    # Hub-stamped at registration. None on construction.
    agent_id: str | None = None
    created_at: str = ""  # ISO-Z, hub-stamped

    def __post_init__(self) -> None:
        # Validate kind at construction so a typo (e.g. ``kind="huMAn"``)
        # fails loudly instead of being silently coerced downstream.
        if self.kind is not None and self.kind not in PASSPORT_KINDS:
            raise ValueError(f"Passport.kind must be one of {PASSPORT_KINDS} or None; got {self.kind!r}")

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

    @property
    def effective_kind(self) -> PassportKind:
        """Resolved ``kind`` — ``None`` falls through to ``"agent"``."""
        return self.kind or "agent"


@dataclass(slots=True)
class ResumeExample:
    title: str
    outcome: str = ""  # "completed" | "failed" | free-form
    task_id: str | None = None
    channel_id: str | None = None
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
    ``observed`` on terminal task events; tenant code may also
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
            payload["examples"] = [ResumeExample(**e) if isinstance(e, dict) else e for e in payload["examples"]]
        if "observed" in payload:
            payload["observed"] = {
                k: ObservedStat(**v) if isinstance(v, dict) else v for k, v in payload["observed"].items()
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
