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

from .audit import (
    AUDIT_KIND_AGENT_REGISTERED,
    AUDIT_KIND_AGENT_UNREGISTERED,
    AUDIT_KIND_EXPECTATION_VIOLATED,
    AUDIT_KIND_RESUME_SET,
    AUDIT_KIND_RULE_SET,
    AUDIT_KIND_SESSION_CLOSED,
    AUDIT_KIND_SESSION_CREATED,
    AUDIT_KIND_SESSION_EXPIRED,
    AUDIT_KIND_SKILL_SET,
    AUDIT_KIND_TASK_TERMINATED,
    RESUME_SOURCE_OBSERVED,
    RESUME_SOURCE_TENANT,
    AuditLog,
)
from .core import Hub
from .expectations import (
    AcksWithinEvaluator,
    AuditHandler,
    AutoCloseHandler,
    ExpectationContext,
    ExpectationEvaluator,
    MaxSilenceEvaluator,
    NotifySessionHandler,
    ReplyWithinEvaluator,
    Violation,
    ViolationHandler,
    default_evaluators,
    default_handlers,
)
from .layout import (
    agents_root,
    audit_path,
    by_capability_path,
    by_name_path,
    inbox_cursor_path,
    inbox_nacks_path,
    inbox_overflow_path,
    passport_path,
    registry_root,
    resume_path,
    rule_path,
    runtime_path,
    session_metadata_path,
    session_tasks_index_path,
    sessions_root,
    skill_path,
    task_events_path,
    task_metadata_path,
    tasks_root,
    wal_path,
)

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
    "AcksWithinEvaluator",
    "AuditHandler",
    "AuditLog",
    "AutoCloseHandler",
    "ExpectationContext",
    "ExpectationEvaluator",
    "Hub",
    "MaxSilenceEvaluator",
    "NotifySessionHandler",
    "RESUME_SOURCE_OBSERVED",
    "RESUME_SOURCE_TENANT",
    "ReplyWithinEvaluator",
    "Violation",
    "ViolationHandler",
    "agents_root",
    "audit_path",
    "by_capability_path",
    "by_name_path",
    "default_evaluators",
    "default_handlers",
    "inbox_cursor_path",
    "inbox_nacks_path",
    "inbox_overflow_path",
    "passport_path",
    "registry_root",
    "resume_path",
    "rule_path",
    "runtime_path",
    "session_metadata_path",
    "session_tasks_index_path",
    "sessions_root",
    "skill_path",
    "task_events_path",
    "task_metadata_path",
    "tasks_root",
    "wal_path",
)
