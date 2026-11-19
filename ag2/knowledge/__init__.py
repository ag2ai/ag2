# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""KnowledgeStore — virtual filesystem for agent knowledge.

A KnowledgeStore provides filesystem semantics over any storage backend.
It stores everything an agent is associated with throughout its lifetime:
operational logs, external artifacts, summaries, and working memory.

Filesystem semantics are used because:
1. LLMs are trained on filesystem operations (read, write, list, delete)
2. Hierarchical paths give free semantic grouping without schema design
3. Any backend (memory, disk, S3, Redis) can implement path-based key-value
"""

from typing import TYPE_CHECKING

from ag2._lazy import LazyExport, lazy_exports
from ag2.exceptions import missing_additional_dependency

from .base import (
    ChangeCallback,
    ChangeSubscription,
    KnowledgeStore,
    NoopChangeSubscription,
)
from .bootstrap import DefaultBootstrap, StoreBootstrap
from .constants import CONVERSATIONS_PREFIX, LOG_PREFIX, WORKING_MEMORY_PATH
from .locked import LockedKnowledgeStore
from .log import EventLogWriter
from .memory import MemoryKnowledgeStore
from .plugin import KnowledgePlugin
from .sqlite import SqliteKnowledgeStore

try:
    from .disk import DiskKnowledgeStore
except ImportError as e:
    DiskKnowledgeStore = missing_additional_dependency("DiskKnowledgeStore", "watchdog>=4.0,<7", e)  # type: ignore[misc]

if TYPE_CHECKING:
    from .redis import RedisKnowledgeStore

# The redis client loads on first use; see docs/adr/0036-provider-exports-load-on-first-use.md.
__getattr__ = lazy_exports(__name__, globals(), {"RedisKnowledgeStore": LazyExport(".redis")})

__all__ = [
    "CONVERSATIONS_PREFIX",
    "LOG_PREFIX",
    "WORKING_MEMORY_PATH",
    "ChangeCallback",
    "ChangeSubscription",
    "DefaultBootstrap",
    "DiskKnowledgeStore",
    "EventLogWriter",
    "KnowledgePlugin",
    "KnowledgeStore",
    "LockedKnowledgeStore",
    "MemoryKnowledgeStore",
    "NoopChangeSubscription",
    "RedisKnowledgeStore",
    "SqliteKnowledgeStore",
    "StoreBootstrap",
]
