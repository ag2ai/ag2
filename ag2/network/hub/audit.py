# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Append-only audit log writer — implemented as a :class:`HubListener`.

Writes a single ``audit.jsonl`` under the hub's ``KnowledgeStore`` root.

The audit log records hub-cross-cutting events that are not visible
on per-channel WALs:

* Identity changes — register, unregister, set_resume (with ``source``:
  ``"tenant"`` for ``set_resume`` calls, ``"observed"`` for
  ``record_observation``), set_skill, set_rule
* Channel lifecycle — created, closed, expired (one record per
  terminal transition)
* Task lifecycle — terminated (completed / failed / expired) for tasks
  the hub observed (mirrored from agent ``Task*`` events)
* Expectation violations — one record per (channel, expectation, violator)
  fire (the sweeper deduplicates so handlers don't re-record)
* Notify-handler crashes — recorded as ``"turn_failed"`` so operators
  can correlate with channel WALs

Each record is a JSON object on its own line with at least
``{"at": ISO-Z, "kind": "<event>"}`` plus event-specific fields.

The audit kind set is **open** — subclasses and tenants may append
records with their own ``kind`` values via :meth:`AuditLog.append`;
the built-in constants below are conveniences for the hub's own
emissions, not a closed enum.

The hub installs ``AuditLog`` as a built-in listener on every newly
opened hub. Tenants do not need to register it. Custom hub subclasses
that want a different audit format can replace it via
:meth:`Hub.replace_audit_log`.
"""

import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

from ag2.knowledge import KnowledgeStore

from .layout import audit_path
from .listener import BaseHubListener


def _default_clock() -> str:
    # ``timezone.utc`` instead of ``datetime.UTC`` (Py3.11+) so the
    # framework keeps its Python 3.10 floor.
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


logger = logging.getLogger(__name__)


AuditSubscriber = Callable[[dict], Awaitable[None]]
ClockFn = Callable[[], str]

__all__ = (
    "AUDIT_KIND_AGENT_REGISTERED",
    "AUDIT_KIND_AGENT_UNREGISTERED",
    "AUDIT_KIND_CHANNEL_CLOSED",
    "AUDIT_KIND_CHANNEL_CREATED",
    "AUDIT_KIND_CHANNEL_EXPIRED",
    "AUDIT_KIND_EXPECTATION_VIOLATED",
    "AUDIT_KIND_RESUME_SET",
    "AUDIT_KIND_RULE_SET",
    "AUDIT_KIND_SKILL_SET",
    "AUDIT_KIND_TASK_TERMINATED",
    "AUDIT_KIND_TURN_FAILED",
    "RESUME_SOURCE_OBSERVED",
    "RESUME_SOURCE_TENANT",
    "AuditLog",
    "AuditSubscriber",
)


AUDIT_KIND_AGENT_REGISTERED = "agent_registered"
AUDIT_KIND_AGENT_UNREGISTERED = "agent_unregistered"
AUDIT_KIND_RESUME_SET = "resume_set"
AUDIT_KIND_RULE_SET = "rule_set"
AUDIT_KIND_SKILL_SET = "skill_set"
AUDIT_KIND_EXPECTATION_VIOLATED = "expectation_violated"
AUDIT_KIND_CHANNEL_CREATED = "channel_created"
AUDIT_KIND_CHANNEL_CLOSED = "channel_closed"
AUDIT_KIND_CHANNEL_EXPIRED = "channel_expired"
AUDIT_KIND_TASK_TERMINATED = "task_terminated"
AUDIT_KIND_TURN_FAILED = "turn_failed"

# ``source`` values for ``resume_set`` audit records.
RESUME_SOURCE_TENANT = "tenant"
RESUME_SOURCE_OBSERVED = "observed"


class AuditLog(BaseHubListener):
    """Append-only writer over the hub's ``KnowledgeStore``.

    Implements :class:`HubListener` so the hub fans out state-transition
    events through the same Protocol every other observer uses. Each
    listener method translates its event into one structured audit
    record via :meth:`append`.

    Subscribers attached via :meth:`subscribe` receive every appended
    record live (in addition to the on-disk append). Subscriber
    exceptions are logged and swallowed — a buggy live tail cannot
    break the persistent log.
    """

    def __init__(self, store: KnowledgeStore, *, clock: ClockFn | None = None) -> None:
        # __init__ stores params; no side effects.
        self._store = store
        self._clock = clock if clock is not None else _default_clock
        self._subscribers: list[AuditSubscriber] = []
        # Running byte counter — process-local, reset on hydrate. Cheap
        # to read for ``Hub.health()`` without touching the store.
        self._bytes_written = 0

    # ── Direct write surface ─────────────────────────────────────────────────

    async def append(self, record: dict) -> None:
        """Serialise and append one record. Notifies subscribers afterwards.

        Public so tenants and hub subclasses can append records with
        custom ``kind`` values that the built-in listener methods
        don't cover.
        """
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        await self._store.append(audit_path(), line)
        self._bytes_written += len(line.encode("utf-8"))
        for subscriber in self._subscribers:
            try:
                await subscriber(record)
            except Exception:
                logger.exception("audit subscriber raised: kind=%s", record.get("kind"))

    @property
    def bytes_written(self) -> int:
        """Process-local byte counter for the audit log.

        Resets on hub restart. Cheap to read — :meth:`Hub.health`
        surfaces this as ``audit_log_bytes`` so operators can graph
        audit volume without touching the store.
        """
        return self._bytes_written

    async def read_all(self) -> list[dict]:
        """Read and parse the entire audit log. Returns ``[]`` if absent."""
        data = await self._store.read(audit_path())
        if not data:
            return []
        records: list[dict] = []
        for line in data.splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records

    def subscribe(self, callback: AuditSubscriber) -> None:
        """Attach a live callback fired per appended record.

        Useful for tailing the audit stream without polling the file —
        e.g. for an operational dashboard or live alert pipeline.
        Callbacks run sequentially in registration order. An exception
        in one callback is logged and does not abort subsequent ones.
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: AuditSubscriber) -> None:
        """Detach a previously-registered subscriber. No-op if absent."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    # ── Listener Protocol impl ───────────────────────────────────────────────

    async def on_agent_event(self, agent_id: str, kind: str, payload: dict) -> None:
        """Translate identity-lifecycle events into audit records.

        Recognises ``"registered"``, ``"unregistered"``, ``"resume_set"``,
        ``"skill_set"``, ``"rule_set"``, and ``"observation_recorded"``.
        Unknown kinds are ignored — subclasses fan out their own kinds
        via :meth:`append` directly.
        """
        at = payload.get("at") or self._clock()
        if kind == "registered":
            passport = payload.get("passport")
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_AGENT_REGISTERED,
                "agent_id": agent_id,
                "name": getattr(passport, "name", None),
            })
        elif kind == "unregistered":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_AGENT_UNREGISTERED,
                "agent_id": agent_id,
                "name": payload.get("name"),
            })
        elif kind == "resume_set":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_RESUME_SET,
                "source": RESUME_SOURCE_TENANT,
                "agent_id": agent_id,
                "version": payload.get("version"),
            })
        elif kind == "skill_set":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_SKILL_SET,
                "agent_id": agent_id,
                "removed": payload.get("removed", False),
            })
        elif kind == "rule_set":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_RULE_SET,
                "agent_id": agent_id,
                "version": payload.get("version"),
            })
        elif kind == "observation_recorded":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_RESUME_SET,
                "source": RESUME_SOURCE_OBSERVED,
                "agent_id": agent_id,
                "version": payload.get("version"),
                "capability": payload.get("capability"),
                "outcome": payload.get("outcome"),
            })

    async def on_channel_event(self, channel_id: str, kind: str, payload: dict) -> None:
        """Translate channel-lifecycle events into audit records.

        Records ``"created"``, ``"closed"``, and ``"expired"``. Other
        kinds (``"opened"``, ``"participant_removed"``,
        ``"participant_hidden"``) are observed but not separately
        audited — they show up on the channel WAL and / or in
        per-violation records.
        """
        at = payload.get("at") or self._clock()
        if kind == "created":
            metadata = payload.get("metadata")
            participants = payload.get("participants")
            if participants is None and metadata is not None:
                participants = [p.agent_id for p in metadata.participants]
            record: dict = {
                "at": at,
                "kind": AUDIT_KIND_CHANNEL_CREATED,
                "channel_id": channel_id,
            }
            if metadata is not None:
                record["manifest_type"] = metadata.manifest.type
                record["manifest_version"] = metadata.manifest.version
                record["creator_id"] = metadata.creator_id
            if participants is not None:
                record["participants"] = list(participants)
            await self.append(record)
        elif kind == "closed":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_CHANNEL_CLOSED,
                "channel_id": channel_id,
                "reason": payload.get("reason"),
            })
        elif kind == "expired":
            await self.append({
                "at": at,
                "kind": AUDIT_KIND_CHANNEL_EXPIRED,
                "channel_id": channel_id,
                "reason": payload.get("reason"),
            })

    async def on_expectation_fired(self, channel_id: str, expectation, violation) -> None:
        """Record one violation per ``(channel, expectation, violator)`` fire."""
        at = self._clock()
        await self.append({
            "at": at,
            "kind": AUDIT_KIND_EXPECTATION_VIOLATED,
            "channel_id": channel_id,
            "expectation": violation.expectation.name,
            "on_violation": violation.expectation.on_violation,
            "params": dict(violation.expectation.params),
            "violators": list(violation.violator_ids),
            "detail": dict(violation.detail),
        })

    async def on_task_event(self, task_id: str, kind: str, payload: dict) -> None:
        """Record terminal task transitions.

        ``"started"`` / ``"progress"`` are observed-only; only the
        terminal kinds (``"completed"`` / ``"failed"`` / ``"expired"`` /
        ``"cancelled"``) become audit records. ``"mirror_failed"``
        signals that a hub-side mirror could not record the agent's
        terminal event — the audit reflects the failure separately.
        """
        if kind not in ("completed", "failed", "expired", "cancelled"):
            return
        at = payload.get("at") or self._clock()
        await self.append({
            "at": at,
            "kind": AUDIT_KIND_TASK_TERMINATED,
            "task_id": task_id,
            "owner_id": payload.get("owner_id"),
            "channel_id": payload.get("channel_id"),
            "outcome": payload.get("outcome", kind),
            "capability": payload.get("capability"),
            "reason": payload.get("reason"),
        })

    async def on_turn_failed(
        self,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:
        """Record a notify-handler crash so operators can correlate with the WAL."""
        await self.append({
            "at": self._clock(),
            "kind": AUDIT_KIND_TURN_FAILED,
            "channel_id": channel_id,
            "agent_id": agent_id,
            "envelope_id": envelope_id,
            "exc_type": type(exc).__name__,
            "exc_message": str(exc),
        })
