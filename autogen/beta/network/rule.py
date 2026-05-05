# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Per-(hub, agent) rules — access + limits.

V1 ships ``access`` + ``limits`` only; transforms (per-envelope local
enforcement) ship in Phase 3 alongside the WebSocket transport.
Defaults are permissive: a freshly registered Agent with no rule
changes can talk to anyone, accept any session type, and has no rate
limit. Apps tighten by passing a non-default ``Rule`` to
``hub_client.register(...)``.

Both blocks are enforced at the **hub**, never the client.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = (
    "AccessBlock",
    "InboxBlock",
    "LimitsBlock",
    "RateBlock",
    "Rule",
    "SessionTypeAccess",
    "parse_duration",
)


_DURATION_UNITS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
}


def parse_duration(s: str | int) -> int:
    """Parse a duration string into seconds.

    Accepts ``"30s"``, ``"15m"``, ``"2h"``, ``"1d"``, plain integer
    strings (treated as seconds), or already-parsed ``int``. Empty
    string returns 0. Raises ``ValueError`` on unknown unit.
    """
    if isinstance(s, int):
        return s
    if not s:
        return 0
    if s[-1] in _DURATION_UNITS:
        unit = s[-1]
        value = s[:-1]
        return int(value) * _DURATION_UNITS[unit]
    return int(s)


@dataclass(slots=True)
class SessionTypeAccess:
    initiate: list[str] = field(default_factory=lambda: ["*"])
    accept: list[str] = field(default_factory=lambda: ["*"])


@dataclass(slots=True)
class AccessBlock:
    inbound_from: list[str] = field(default_factory=lambda: ["*"])  # globs over `name`
    outbound_to: list[str] = field(default_factory=lambda: ["*"])
    session_types: SessionTypeAccess = field(default_factory=SessionTypeAccess)


@dataclass(slots=True)
class RateBlock:
    """Token-bucket rate limiter (Phase 2). M1 stores the values but
    does not enforce — ``per_minute = 0`` keeps the limiter disabled
    by default, so the no-op behaviour matches the eventual default.
    """

    per_minute: int = 0
    burst: int = 0


@dataclass(slots=True)
class InboxBlock:
    """Inbox capacity policy.

    M1 ships ``reject`` overflow only; ``drop_oldest`` and
    ``drop_newest`` arrive in Phase 2.
    """

    max_pending: int = 1000
    overflow: str = "reject"  # "reject" | "drop_oldest" | "drop_newest"


@dataclass(slots=True)
class LimitsBlock:
    """Concurrency caps + parsed duration TTLs + failure-mode thresholds.

    ``0`` disables a numeric cap. Duration strings are parsed via
    :func:`parse_duration`; values may be passed pre-parsed as ``int``
    seconds.
    """

    max_concurrent_sessions: int = 0
    max_concurrent_tasks: int = 0
    session_ttl_default: str = "2h"
    task_ttl_default: str = "15m"
    rate: RateBlock = field(default_factory=RateBlock)
    delegation_depth: int = 5
    inbox: InboxBlock = field(default_factory=InboxBlock)

    # Failure-mode thresholds (M3 sweepers honour these)
    peer_heartbeat_timeout: str = "30s"
    task_stall_threshold: str = "60s"
    session_idle_threshold: str = "5m"


@dataclass(slots=True)
class Rule:
    version: int = 1
    access: AccessBlock = field(default_factory=AccessBlock)
    limits: LimitsBlock = field(default_factory=LimitsBlock)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Rule":
        payload = dict(data)
        access = payload.get("access")
        if isinstance(access, dict):
            access_payload = dict(access)
            session_types = access_payload.get("session_types")
            if isinstance(session_types, dict):
                access_payload["session_types"] = SessionTypeAccess(**session_types)
            payload["access"] = AccessBlock(**access_payload)
        limits = payload.get("limits")
        if isinstance(limits, dict):
            limits_payload = dict(limits)
            rate = limits_payload.get("rate")
            if isinstance(rate, dict):
                limits_payload["rate"] = RateBlock(**rate)
            inbox = limits_payload.get("inbox")
            if isinstance(inbox, dict):
                limits_payload["inbox"] = InboxBlock(**inbox)
            payload["limits"] = LimitsBlock(**limits_payload)
        return cls(**payload)
