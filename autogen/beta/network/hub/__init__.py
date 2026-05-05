# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub — registry, dispatcher, and state-machine owner.

The hub is the only place that has cross-tenant visibility. It owns
the registry, session and task state machines, the WAL, the dispatch
path, the adapter state cache, and the internal sweepers. It never
calls ``Agent.ask``, executes tenant transforms, or imports tenant
modules — the trust boundary runs through ``HubClient`` /
``AgentClient`` (see ``client/``).
"""

from .core import Hub
from .layout import (
    agents_root,
    audit_path,
    by_capability_path,
    by_name_path,
    passport_path,
    resume_path,
    rule_path,
    runtime_path,
    session_metadata_path,
    sessions_root,
    skill_path,
    task_metadata_path,
    tasks_root,
    wal_path,
)

__all__ = (
    "Hub",
    "agents_root",
    "audit_path",
    "by_capability_path",
    "by_name_path",
    "passport_path",
    "resume_path",
    "rule_path",
    "runtime_path",
    "session_metadata_path",
    "sessions_root",
    "skill_path",
    "task_metadata_path",
    "tasks_root",
    "wal_path",
)
