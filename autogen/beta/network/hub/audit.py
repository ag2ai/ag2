# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Append-only audit log writer.

Writes to a single ``audit.jsonl`` under the hub's ``KnowledgeStore``
root.

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

Each record is a JSON object on its own line with at least
``{"at": ISO-Z, "kind": "<event>"}`` plus event-specific fields.

The audit kind set is **open** — subclasses and tenants may append
records with their own ``kind`` values; the built-in constants below
are conveniences for the hub's own emissions, not a closed enum.
"""

import contextlib
import json
import logging
from collections.abc import Awaitable, Callable

from autogen.beta.knowledge import KnowledgeStore

from .layout import audit_path

logger = logging.getLogger(__name__)


AuditSubscriber = Callable[[dict], Awaitable[None]]

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

# ``source`` values for ``resume_set`` audit records.
RESUME_SOURCE_TENANT = "tenant"
RESUME_SOURCE_OBSERVED = "observed"


class AuditLog:
    """Append-only writer over the hub's ``KnowledgeStore``.

    Stateless — every ``append`` is one JSON line. Reads are O(file
    size) and intended for tests / admin tooling, not hot paths.

    Subscribers attached via :meth:`subscribe` receive every appended
    record live (in addition to the on-disk append). Subscriber
    exceptions are logged and swallowed — a buggy live tail cannot
    break the persistent log.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        # __init__ stores params; no side effects.
        self._store = store
        self._subscribers: list[AuditSubscriber] = []

    async def append(self, record: dict) -> None:
        """Serialise and append one record. Notifies subscribers afterwards."""
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        await self._store.append(audit_path(), line)
        for subscriber in self._subscribers:
            try:
                await subscriber(record)
            except Exception:
                logger.exception("audit subscriber raised: kind=%s", record.get("kind"))

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
