# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""KnowledgeStore — virtual filesystem for actor knowledge.

A KnowledgeStore provides filesystem semantics over any storage backend.
It stores everything an actor is associated with throughout its lifetime:
operational logs, external artifacts, summaries, and working memory.

Filesystem semantics are used because:
1. LLMs are trained on filesystem operations (read, write, list, delete)
2. Hierarchical paths give free semantic grouping without schema design
3. Any backend (memory, disk, S3, Redis) can implement path-based key-value
"""

from autogen.beta.exceptions import missing_optional_dependency

from .base import (
    ChangeCallback,
    ChangeSubscription,
    KnowledgeStore,
    NoopChangeSubscription,
)
from .bootstrap import DefaultBootstrap, StoreBootstrap
from .locked import LockedKnowledgeStore
from .log import EventLogWriter
from .memory import MemoryKnowledgeStore
from .redis import RedisKnowledgeStore
from .sqlite import SqliteKnowledgeStore

try:
    from .disk import DiskKnowledgeStore
except ImportError as e:
    DiskKnowledgeStore = missing_optional_dependency("DiskKnowledgeStore", "watchdog", e)  # type: ignore[misc]

__all__ = [
    "ChangeCallback",
    "ChangeSubscription",
    "DefaultBootstrap",
    "DiskKnowledgeStore",
    "EventLogWriter",
    "KnowledgeStore",
    "LockedKnowledgeStore",
    "MemoryKnowledgeStore",
    "NoopChangeSubscription",
    "RedisKnowledgeStore",
    "SqliteKnowledgeStore",
    "StoreBootstrap",
]
