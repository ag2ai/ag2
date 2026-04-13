# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Rule — declarative per-actor policy.

A :class:`Rule` is a three-layer dataclass: **access** (who may talk to whom
and over which session types), **limits** (cross-call budgets and caps), and
**transforms** (per-envelope interception hooks). Phase 1 enforces only the
access layer plus ``limits.max_concurrent_sessions`` and ``session_ttl_default``;
the transforms list is stored verbatim but not executed. Subsequent phases
layer on the remaining limits and the five transform apply forms.

All fields default to permissive values so a minimally-specified rule just
works for development. Patterns in ``inbound_from`` / ``outbound_to`` / etc.
use simple glob matching (``*`` wildcards, ``:`` segment separator).
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any

from .session_types import SessionType


def _default_session_types() -> list[str]:
    return [t.value for t in SessionType]


# ---------------------------------------------------------------------------
# Access block
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SessionTypeAccess:
    initiate: list[str] = field(default_factory=_default_session_types)
    accept: list[str] = field(default_factory=_default_session_types)

    def may_initiate(self, session_type: SessionType | str) -> bool:
        return _match(_session_type_value(session_type), self.initiate)

    def may_accept(self, session_type: SessionType | str) -> bool:
        return _match(_session_type_value(session_type), self.accept)

    def to_dict(self) -> dict[str, Any]:
        return {"initiate": list(self.initiate), "accept": list(self.accept)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTypeAccess:
        return cls(
            initiate=list(data.get("initiate", _default_session_types())),
            accept=list(data.get("accept", _default_session_types())),
        )


# ---------------------------------------------------------------------------
# Subscription access policy
# ---------------------------------------------------------------------------


SUBSCRIBE_MEMBER_ONLY = "member-only"
SUBSCRIBE_HUB_PUBLIC = "public-within-hub"
SUBSCRIBE_PUBLIC = "public"

_VALID_SUBSCRIBE_POLICIES: frozenset[str] = frozenset(
    {SUBSCRIBE_MEMBER_ONLY, SUBSCRIBE_HUB_PUBLIC, SUBSCRIBE_PUBLIC}
)


@dataclass(slots=True)
class SubscribeAccess:
    """Who may open non-participant subscriptions on sessions / tasks.

    ``sessions``:

    * ``member-only`` (default) — only session participants may subscribe.
    * ``public-within-hub`` — any actor registered with the hub may
      subscribe.
    * ``public`` — reserved for federation (Phase 7); currently behaves
      the same as ``public-within-hub`` locally.

    ``tasks`` is reserved for Phase 4 task subscriptions; Phase 2 only
    uses ``sessions``.
    """

    sessions: str = SUBSCRIBE_MEMBER_ONLY
    tasks: str = "owner-or-member"

    def allows_session_observer(
        self, *, is_participant: bool, is_hub_member: bool
    ) -> bool:
        if is_participant:
            return True
        if self.sessions == SUBSCRIBE_MEMBER_ONLY:
            return False
        if self.sessions in (SUBSCRIBE_HUB_PUBLIC, SUBSCRIBE_PUBLIC):
            return is_hub_member
        return False

    def to_dict(self) -> dict[str, Any]:
        return {"sessions": self.sessions, "tasks": self.tasks}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubscribeAccess:
        sessions = str(data.get("sessions", SUBSCRIBE_MEMBER_ONLY))
        if sessions not in _VALID_SUBSCRIBE_POLICIES:
            raise ValueError(
                f"access.subscribe.sessions must be one of "
                f"{sorted(_VALID_SUBSCRIBE_POLICIES)}, got {sessions!r}"
            )
        return cls(
            sessions=sessions,
            tasks=str(data.get("tasks", "owner-or-member")),
        )


@dataclass(slots=True)
class AccessBlock:
    """Who may interact with this actor."""

    inbound_from: list[str] = field(default_factory=lambda: ["*"])
    outbound_to: list[str] = field(default_factory=lambda: ["*"])
    session_types: SessionTypeAccess = field(default_factory=SessionTypeAccess)
    subscribe: SubscribeAccess = field(default_factory=SubscribeAccess)

    def allows_inbound(self, sender_name: str) -> bool:
        return _match(sender_name, self.inbound_from)

    def allows_outbound(self, recipient_name: str) -> bool:
        return _match(recipient_name, self.outbound_to)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inbound_from": list(self.inbound_from),
            "outbound_to": list(self.outbound_to),
            "session_types": self.session_types.to_dict(),
            "subscribe": self.subscribe.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessBlock:
        return cls(
            inbound_from=list(data.get("inbound_from", ["*"])),
            outbound_to=list(data.get("outbound_to", ["*"])),
            session_types=SessionTypeAccess.from_dict(data.get("session_types", {})),
            subscribe=SubscribeAccess.from_dict(data.get("subscribe", {})),
        )


# ---------------------------------------------------------------------------
# Limits block
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RateBlock:
    """Rate-limit parameters for a token-bucket enforcement path.

    ``per_minute`` is the refill rate (envelopes per minute).
    ``burst`` is the maximum bucket capacity — how many envelopes can
    queue up in a sudden spike before being throttled. Setting
    ``per_minute`` to zero disables rate limiting; ``burst`` defaults
    to ``per_minute`` when not supplied so a small steady-state limit
    doesn't accidentally paper over a 10x burst.
    """

    per_minute: int = 0  # 0 disables
    burst: int = 0

    def is_enabled(self) -> bool:
        return self.per_minute > 0

    def effective_burst(self) -> int:
        return self.burst if self.burst > 0 else self.per_minute

    def to_dict(self) -> dict[str, Any]:
        return {"per_minute": self.per_minute, "burst": self.burst}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RateBlock:
        return cls(
            per_minute=int(data.get("per_minute", 0)),
            burst=int(data.get("burst", 0)),
        )


@dataclass(slots=True)
class LimitsBlock:
    max_concurrent_sessions: int = 32
    max_concurrent_tasks: int = 64
    session_ttl_default: str = "2h"
    task_ttl_default: str = "15m"
    delegation_depth: int = 5
    rate: RateBlock = field(default_factory=RateBlock)
    # Phase 2 stores these two verbatim; per-hour and per-day window
    # enforcement lands with Phase 4 (tasks), where attribution to a
    # specific LLM call is natural.
    tokens_per_hour: int = 0  # 0 disables
    cost_per_day_usd: float = 0.0  # 0 disables

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_concurrent_sessions": self.max_concurrent_sessions,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "session_ttl_default": self.session_ttl_default,
            "task_ttl_default": self.task_ttl_default,
            "delegation_depth": self.delegation_depth,
            "rate": self.rate.to_dict(),
            "tokens_per_hour": self.tokens_per_hour,
            "cost_per_day_usd": self.cost_per_day_usd,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LimitsBlock:
        return cls(
            max_concurrent_sessions=int(data.get("max_concurrent_sessions", 32)),
            max_concurrent_tasks=int(data.get("max_concurrent_tasks", 64)),
            session_ttl_default=str(data.get("session_ttl_default", "2h")),
            task_ttl_default=str(data.get("task_ttl_default", "15m")),
            delegation_depth=int(data.get("delegation_depth", 5)),
            rate=RateBlock.from_dict(data.get("rate", {})),
            tokens_per_hour=int(data.get("tokens_per_hour", 0)),
            cost_per_day_usd=float(data.get("cost_per_day_usd", 0.0)),
        )

    def session_ttl_seconds(self) -> int:
        return parse_duration(self.session_ttl_default)

    def task_ttl_seconds(self) -> int:
        return parse_duration(self.task_ttl_default)


# ---------------------------------------------------------------------------
# Transforms (stored verbatim in Phase 1; enforced in Phase 5)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TransformSpec:
    stage: str
    apply: Any
    when: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "apply": self.apply, "when": dict(self.when)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformSpec:
        return cls(
            stage=data["stage"],
            apply=data.get("apply"),
            when=dict(data.get("when", {})),
        )


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Rule:
    """Per-actor rule bundle (access / limits / transforms)."""

    version: int = 1
    access: AccessBlock = field(default_factory=AccessBlock)
    limits: LimitsBlock = field(default_factory=LimitsBlock)
    transforms: list[TransformSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "access": self.access.to_dict(),
            "limits": self.limits.to_dict(),
            "transforms": [t.to_dict() for t in self.transforms],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        return cls(
            version=int(data.get("version", 1)),
            access=AccessBlock.from_dict(data.get("access", {})),
            limits=LimitsBlock.from_dict(data.get("limits", {})),
            transforms=[TransformSpec.from_dict(t) for t in data.get("transforms", [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Rule:
        return cls.from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _match(value: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)


def _session_type_value(session_type: SessionType | str) -> str:
    return session_type.value if isinstance(session_type, SessionType) else session_type


_DURATION_UNITS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def parse_duration(value: str) -> int:
    """Parse a compact duration string (``"2h"``, ``"15m"``, ``"30s"``) to seconds.

    Accepts a trailing unit letter from ``s`` / ``m`` / ``h`` / ``d``. A bare
    integer string is treated as seconds.
    """

    text = value.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():
        return int(text)
    unit = text[-1]
    if unit not in _DURATION_UNITS:
        raise ValueError(f"unknown duration unit {unit!r} in {value!r}")
    number = text[:-1]
    if not number.isdigit():
        raise ValueError(f"invalid duration {value!r}")
    return int(number) * _DURATION_UNITS[unit]
