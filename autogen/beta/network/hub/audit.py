# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Append-only audit log writer.

Writes to a single ``audit.jsonl`` under the hub's ``KnowledgeStore``
root.

The audit log records hub-cross-cutting events that are not visible
on per-session WALs:

* Identity changes — register, unregister, set_resume (with ``source``:
  ``"tenant"`` for ``set_resume`` calls, ``"observed"`` for
  ``record_observation``), set_skill, set_rule
* Session lifecycle — created, closed, expired (one record per
  terminal transition)
* Task lifecycle — terminated (completed / failed / expired) for tasks
  the hub observed (mirrored from agent ``Task*`` events)
* Expectation violations — one record per (session, expectation, violator)
  fire (the sweeper deduplicates so handlers don't re-record)

Each record is a JSON object on its own line with at least
``{"at": ISO-Z, "kind": "<event>"}`` plus event-specific fields.
"""

import json

from autogen.beta.knowledge import KnowledgeStore

from .layout import audit_path

__all__ = (
    "AUDIT_KIND_AGENT_REGISTERED",
    "AUDIT_KIND_AGENT_UNREGISTERED",
    "AUDIT_KIND_EXPECTATION_VIOLATED",
    "AUDIT_KIND_RESUME_SET",
    "AUDIT_KIND_RULE_SET",
    "AUDIT_KIND_SESSION_CLOSED",
    "AUDIT_KIND_SESSION_CREATED",
    "AUDIT_KIND_SESSION_EXPIRED",
    "AUDIT_KIND_SKILL_SET",
    "AUDIT_KIND_TASK_TERMINATED",
    "RESUME_SOURCE_OBSERVED",
    "RESUME_SOURCE_TENANT",
    "AuditLog",
)


AUDIT_KIND_AGENT_REGISTERED = "agent_registered"
AUDIT_KIND_AGENT_UNREGISTERED = "agent_unregistered"
AUDIT_KIND_RESUME_SET = "resume_set"
AUDIT_KIND_RULE_SET = "rule_set"
AUDIT_KIND_SKILL_SET = "skill_set"
AUDIT_KIND_EXPECTATION_VIOLATED = "expectation_violated"
AUDIT_KIND_SESSION_CREATED = "session_created"
AUDIT_KIND_SESSION_CLOSED = "session_closed"
AUDIT_KIND_SESSION_EXPIRED = "session_expired"
AUDIT_KIND_TASK_TERMINATED = "task_terminated"

# ``source`` values for ``resume_set`` audit records.
RESUME_SOURCE_TENANT = "tenant"
RESUME_SOURCE_OBSERVED = "observed"


class AuditLog:
    """Append-only writer over the hub's ``KnowledgeStore``.

    Stateless — every ``append`` is one JSON line. Reads are O(file
    size) and intended for tests / admin tooling, not hot paths.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        # __init__ stores params; no side effects.
        self._store = store

    async def append(self, record: dict) -> None:
        """Serialise and append one record."""
        line = json.dumps(record, default=str, sort_keys=True) + "\n"
        await self._store.append(audit_path(), line)

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
