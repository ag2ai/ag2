# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""FS layout constants for the hub.

The hub writes everything under a ``hub/`` prefix on its backing
KnowledgeStore so a shared store can host both actor-private knowledge
(``actor/...``) and hub state without collision.
"""

from __future__ import annotations

HUB_ROOT = "/hub"
HUB_CONFIG = f"{HUB_ROOT}/config.json"

ACTORS_ROOT = f"{HUB_ROOT}/actors"
SESSIONS_ROOT = f"{HUB_ROOT}/sessions"
NAME_INDEX = f"{HUB_ROOT}/registry/by_name"


def actor_dir(actor_id: str) -> str:
    return f"{ACTORS_ROOT}/{actor_id}"


def actor_identity(actor_id: str) -> str:
    return f"{actor_dir(actor_id)}/identity.json"


def actor_rule(actor_id: str) -> str:
    return f"{actor_dir(actor_id)}/rule.json"


def actor_skill(actor_id: str) -> str:
    return f"{actor_dir(actor_id)}/SKILL.md"


def actor_runtime(actor_id: str) -> str:
    return f"{actor_dir(actor_id)}/runtime.json"


def actor_inbox_dir(actor_id: str) -> str:
    return f"{actor_dir(actor_id)}/inbox"


def actor_inbox_pending_dir(actor_id: str) -> str:
    return f"{actor_inbox_dir(actor_id)}/pending"


def actor_inbox_received_dir(actor_id: str) -> str:
    return f"{actor_inbox_dir(actor_id)}/received"


def actor_inbox_overflow_dir(actor_id: str) -> str:
    return f"{actor_inbox_dir(actor_id)}/overflow"


def actor_inbox_pending(actor_id: str, envelope_id: str) -> str:
    return f"{actor_inbox_pending_dir(actor_id)}/{envelope_id}.json"


def actor_inbox_received(actor_id: str, envelope_id: str) -> str:
    return f"{actor_inbox_received_dir(actor_id)}/{envelope_id}.json"


def actor_inbox_overflow(actor_id: str, envelope_id: str) -> str:
    return f"{actor_inbox_overflow_dir(actor_id)}/{envelope_id}.json"


def actor_inbox_nacks(actor_id: str) -> str:
    return f"{actor_inbox_dir(actor_id)}/nacks.jsonl"


def name_pointer(name: str) -> str:
    # Store one file per registered name that points to the actor_id.
    # The filename is the sha-safe version of the name; we keep ":" because
    # backends accept it on macOS + Linux.
    return f"{NAME_INDEX}/{name}.txt"


def session_dir(session_id: str) -> str:
    return f"{SESSIONS_ROOT}/{session_id}"


def session_metadata(session_id: str) -> str:
    return f"{session_dir(session_id)}/metadata.json"


def session_wal(session_id: str) -> str:
    return f"{session_dir(session_id)}/wal.jsonl"


def hub_config() -> str:
    return HUB_CONFIG
