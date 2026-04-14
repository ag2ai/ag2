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
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .session_types import SessionType


def _default_session_types() -> list[str]:
    return [t.value for t in SessionType]


class TransformStage(str, Enum):
    """The four pipeline stages a :class:`TransformSpec` can declare.

    Keeping this next to :class:`TransformSpec` (rather than inside
    ``client/transforms/``) lets :meth:`Rule.from_dict` validate the
    stage field without importing client-runtime code — rules parse
    identically on the hub side and on actor-local test fixtures.

    Subclassing ``str`` lets members compare equal to their literal
    value, so existing call sites that say ``spec.stage == "pre_send"``
    keep working after the Phase 5a.1 validation upgrade.
    """

    PRE_SEND = "pre_send"
    POST_SEND = "post_send"
    PRE_RECEIVE = "pre_receive"
    POST_RECEIVE = "post_receive"


_VALID_STAGES = frozenset(stage.value for stage in TransformStage)


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


# ---------------------------------------------------------------------------
# Knowledge access policy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KnowledgeAccess:
    """Read-only cross-actor access to the owning actor's KnowledgeStore.

    Every actor owns a private :class:`KnowledgeStore` that is otherwise
    invisible to every other actor. ``KnowledgeAccess`` carves out a
    read-only slice of that store for specific peers to read via
    ``GET /v1/actors/{id}/knowledge/{path}``:

    * ``expose`` — glob patterns matched against the requested path in
      the owning actor's store. Patterns use :mod:`fnmatch` syntax (``*``,
      ``**``, ``?``). Defaults to the empty list — no exposure.
    * ``readers`` — glob patterns matched against the *requesting* actor's
      ``identity.name``. Defaults to the empty list — no reader is allowed.

    A read is permitted iff **both** the requesting actor's name matches
    some ``readers`` entry **and** the requested path matches some
    ``expose`` entry. Empty lists deny by default, matching the V2
    ``_exposed_paths`` side-channel replacement called out in §12.

    Writes are never exposed — the endpoint only serves ``read``.
    """

    expose: list[str] = field(default_factory=list)
    readers: list[str] = field(default_factory=list)

    def allows(self, *, reader_name: str, path: str) -> bool:
        if not self.expose or not self.readers:
            return False
        if not _match(reader_name, self.readers):
            return False
        return _match_path(path, self.expose)

    def to_dict(self) -> dict[str, Any]:
        return {"expose": list(self.expose), "readers": list(self.readers)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeAccess:
        return cls(
            expose=list(data.get("expose", [])),
            readers=list(data.get("readers", [])),
        )


@dataclass(slots=True)
class AccessBlock:
    """Who may interact with this actor."""

    inbound_from: list[str] = field(default_factory=lambda: ["*"])
    outbound_to: list[str] = field(default_factory=lambda: ["*"])
    session_types: SessionTypeAccess = field(default_factory=SessionTypeAccess)
    subscribe: SubscribeAccess = field(default_factory=SubscribeAccess)
    knowledge: KnowledgeAccess = field(default_factory=KnowledgeAccess)

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
            "knowledge": self.knowledge.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AccessBlock:
        return cls(
            inbound_from=list(data.get("inbound_from", ["*"])),
            outbound_to=list(data.get("outbound_to", ["*"])),
            session_types=SessionTypeAccess.from_dict(data.get("session_types", {})),
            subscribe=SubscribeAccess.from_dict(data.get("subscribe", {})),
            knowledge=KnowledgeAccess.from_dict(data.get("knowledge", {})),
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


# ---------------------------------------------------------------------------
# Inbox block — Phase 3a §7.1 compliance
# ---------------------------------------------------------------------------


INBOX_OVERFLOW_REJECT = "reject"
INBOX_OVERFLOW_SPOOL = "spool"
INBOX_OVERFLOW_DROP_OLDEST = "drop_oldest"
INBOX_OVERFLOW_DROP_NEWEST = "drop_newest"

_VALID_INBOX_OVERFLOW: frozenset[str] = frozenset(
    {
        INBOX_OVERFLOW_REJECT,
        INBOX_OVERFLOW_SPOOL,
        INBOX_OVERFLOW_DROP_OLDEST,
        INBOX_OVERFLOW_DROP_NEWEST,
    }
)


@dataclass(slots=True)
class InboxBlock:
    """Per-actor inbox configuration.

    ``max_pending`` is the ceiling on how many envelopes may sit in the
    actor's ``hub/actors/{id}/inbox/pending/`` directory before the
    overflow policy kicks in. ``0`` disables the check (unlimited).

    ``overflow`` selects the back-pressure mode:

    * ``reject`` (default) — synchronously raise :class:`InboxFullError`
      on ``post_envelope``. The sender sees the error on its ``send``
      frame and can retry with backoff. Matches design §13.7.
    * ``spool`` — write to ``hub/actors/{id}/inbox/overflow/`` instead
      of ``pending/``. Spooled envelopes are not counted against
      ``max_pending`` and are NOT pushed via ``notify`` (the actor
      drains them on reconnect or via an explicit admin call).
    * ``drop_oldest`` / ``drop_newest`` — evict the oldest / newest
      pending envelope to make room. **Ship in Phase 3b** — Phase 3a
      stores the policy name verbatim and rejects unknown values on
      ``from_dict``.

    Phase 3a enforces ``reject`` and ``spool`` at
    :meth:`Hub.post_envelope` pre-check time (before WAL append so the
    delivery is atomic). ``drop_oldest`` / ``drop_newest`` currently
    fall through to ``reject`` with a warning — authors can ship rules
    that reference them, but the behavior won't change until Phase 3b.
    """

    max_pending: int = 0  # 0 disables
    overflow: str = INBOX_OVERFLOW_REJECT

    def to_dict(self) -> dict[str, Any]:
        return {"max_pending": self.max_pending, "overflow": self.overflow}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InboxBlock:
        overflow = str(data.get("overflow", INBOX_OVERFLOW_REJECT))
        if overflow not in _VALID_INBOX_OVERFLOW:
            raise ValueError(
                f"inbox.overflow must be one of {sorted(_VALID_INBOX_OVERFLOW)}, "
                f"got {overflow!r}"
            )
        return cls(
            max_pending=int(data.get("max_pending", 0)),
            overflow=overflow,
        )


@dataclass(slots=True)
class LimitsBlock:
    max_concurrent_sessions: int = 32
    max_concurrent_tasks: int = 64
    session_ttl_default: str = "2h"
    task_ttl_default: str = "15m"
    delegation_depth: int = 5
    rate: RateBlock = field(default_factory=RateBlock)
    inbox: InboxBlock = field(default_factory=InboxBlock)
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
            "inbox": self.inbox.to_dict(),
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
            inbox=InboxBlock.from_dict(data.get("inbox", {})),
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
    """One transform declaration from a rule's ``transforms`` list.

    Phase 5a.1 tightens stage validation: ``stage`` must be one of the
    four :class:`TransformStage` values. Unknown stage strings raise
    :class:`ValueError` at construction and at :meth:`from_dict`, so
    a typo in a rule upload surfaces at rule-write time rather than
    silently disabling the transform. ``apply`` stays loose
    (``Any``) — the adapter-level shape checks happen in the pipeline
    compiler, and unknown ``apply`` forms (``exec`` / ``ws``) log and
    pass through so forward-compatibility with Phase 5b holds.
    """

    stage: str
    apply: Any
    when: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in _VALID_STAGES:
            raise ValueError(
                f"TransformSpec.stage must be one of {sorted(_VALID_STAGES)}, "
                f"got {self.stage!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "apply": self.apply, "when": dict(self.when)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformSpec:
        try:
            stage = data["stage"]
        except KeyError as exc:
            raise ValueError("TransformSpec requires a 'stage' field") from exc
        return cls(
            stage=stage,
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


def _normalize_store_path(path: str) -> str:
    """Collapse double slashes and ensure a single leading slash."""

    if not path.startswith("/"):
        path = "/" + path
    while "//" in path:
        path = path.replace("//", "/")
    return path


def _path_glob_match(path: str, pattern: str) -> bool:
    """Glob-match a KnowledgeStore path against one pattern.

    Supported forms:

    * Exact path — ``/foo/bar.txt``.
    * Directory prefix — ``/foo/`` matches ``/foo`` and anything beneath.
    * ``*`` — matches a single path segment (no ``/``).
    * ``**`` — matches any number of path segments (including zero).
    * ``?`` — matches one non-``/`` character.
    """

    if pattern.endswith("/"):
        prefix = _normalize_store_path(pattern)
        trimmed = prefix.rstrip("/") or "/"
        return path == trimmed or path.startswith(prefix)
    norm_pattern = _normalize_store_path(pattern)
    parts: list[str] = []
    i = 0
    while i < len(norm_pattern):
        c = norm_pattern[i]
        if c == "*":
            if i + 1 < len(norm_pattern) and norm_pattern[i + 1] == "*":
                parts.append(".*")
                i += 2
            else:
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    regex = "^" + "".join(parts) + "$"
    return re.match(regex, path) is not None


def _match_path(path: str, patterns: list[str]) -> bool:
    """Match a KnowledgeStore path against a list of glob-style patterns."""

    if not patterns:
        return False
    norm = _normalize_store_path(path)
    return any(_path_glob_match(norm, p) for p in patterns)


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
