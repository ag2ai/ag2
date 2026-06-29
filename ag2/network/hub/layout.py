# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Filesystem path helpers for the hub's ``KnowledgeStore`` layout.

All paths are Unix-style with a leading ``/`` (the ``KnowledgeStore``
Protocol normalises internally). Centralising path construction here
means changes to the on-disk layout (chunked WALs, namespace
migration) only touch this module.
"""

__all__ = (
    "agents_root",
    "audit_path",
    "by_capability_path",
    "by_name_path",
    "channel_metadata_path",
    "channel_tasks_index_path",
    "channels_root",
    "inbox_cursor_path",
    "inbox_nacks_path",
    "inbox_overflow_path",
    "passport_path",
    "registry_root",
    "resume_path",
    "rule_path",
    "runtime_path",
    "skill_path",
    "task_checkpoint_path",
    "task_events_path",
    "task_metadata_path",
    "tasks_root",
    "wal_path",
)


# ── Agent files ──────────────────────────────────────────────────────────────


def agents_root() -> str:
    return "/agents"


def passport_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/passport.json"


def resume_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/resume.json"


def skill_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/SKILL.md"


def runtime_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/runtime.json"


def rule_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/rule.json"


def inbox_cursor_path(agent_id: str) -> str:
    """Per-agent JSON map ``{channel_id: last-acked envelope_id}``.

    One cursor per (agent, channel) so an ack in one channel never
    advances the high-water mark of another."""
    return f"/agents/{agent_id}/inbox.cursors.json"


def inbox_nacks_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/inbox_nacks.jsonl"


def inbox_overflow_path(agent_id: str) -> str:
    return f"/agents/{agent_id}/inbox_overflow.jsonl"


# ── Registry caches ──────────────────────────────────────────────────────────


def registry_root() -> str:
    return "/registry"


def by_name_path() -> str:
    return "/registry/by_name.json"


def by_capability_path() -> str:
    return "/registry/by_capability.json"


# ── Channels ─────────────────────────────────────────────────────────────────


def channels_root() -> str:
    return "/channels"


def channel_metadata_path(channel_id: str) -> str:
    return f"/channels/{channel_id}/metadata.json"


def wal_path(channel_id: str) -> str:
    return f"/channels/{channel_id}/wal.jsonl"


def channel_tasks_index_path(channel_id: str) -> str:
    return f"/channels/{channel_id}/tasks.json"


# ── Tasks ────────────────────────────────────────────────────────────────────


def tasks_root() -> str:
    return "/tasks"


def task_metadata_path(task_id: str) -> str:
    return f"/tasks/{task_id}/metadata.json"


def task_events_path(task_id: str) -> str:
    return f"/tasks/{task_id}/events.jsonl"


def task_checkpoint_path(task_id: str) -> str:
    """Owner-supplied resume state for crash recovery. Single JSON
    blob, last-write-wins. Opaque to the framework."""
    return f"/tasks/{task_id}/checkpoint.json"


# ── Audit ────────────────────────────────────────────────────────────────────


def audit_path() -> str:
    """Single append-only audit log."""
    return "/audit/audit.jsonl"
