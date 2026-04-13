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
from dataclasses import asdict, dataclass, field
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


@dataclass(slots=True)
class AccessBlock:
    """Who may interact with this actor."""

    inbound_from: list[str] = field(default_factory=lambda: ["*"])
    outbound_to: list[str] = field(default_factory=lambda: ["*"])
    session_types: SessionTypeAccess = field(default_factory=SessionTypeAccess)

    def allows_inbound(self, sender_name: str) -> bool:
        return _match(sender_name, self.inbound_from)

    def allows_outbound(self, recipient_name: str) -> bool:
        return _match(recipient_name, self.outbound_to)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inbound_from": list(self.inbound_from),
            "outbound_to": list(self.outbound_to),
            "session_types": self.session_types.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessBlock:
        return cls(
            inbound_from=list(data.get("inbound_from", ["*"])),
            outbound_to=list(data.get("outbound_to", ["*"])),
            session_types=SessionTypeAccess.from_dict(data.get("session_types", {})),
        )


# ---------------------------------------------------------------------------
# Limits block
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LimitsBlock:
    max_concurrent_sessions: int = 32
    max_concurrent_tasks: int = 64
    session_ttl_default: str = "2h"
    task_ttl_default: str = "15m"
    delegation_depth: int = 5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LimitsBlock:
        return cls(
            max_concurrent_sessions=int(data.get("max_concurrent_sessions", 32)),
            max_concurrent_tasks=int(data.get("max_concurrent_tasks", 64)),
            session_ttl_default=str(data.get("session_ttl_default", "2h")),
            task_ttl_default=str(data.get("task_ttl_default", "15m")),
            delegation_depth=int(data.get("delegation_depth", 5)),
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
