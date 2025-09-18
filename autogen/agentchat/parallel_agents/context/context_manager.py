# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import threading
from typing import Any

__all__ = ["ContextManager"]


class ContextManager:
    """Manages context variables across parallel processes with real-time synchronization"""

    def __init__(self):
        # Use multiprocessing Manager for shared state across processes
        self.manager = mp.Manager()
        self.shared_context = self.manager.dict()
        self.context_version = mp.Value("i", 0)
        self.context_lock = mp.Lock()

        # Local cache for each process to reduce IPC overhead
        self.local_cache: dict[str, Any] = {}
        self.cache_version = 0
        self.cache_lock = threading.RLock()

        print("ContextManager initialized with shared memory")

    def update_context(self, key: str, value: Any, process_id: int = 0):
        """Update shared dict and increment version for real-time sync"""
        with self.context_lock:
            # Update shared context
            self.shared_context[key] = value

            # Increment version to signal change
            self.context_version.value += 1

            # Update local cache
            with self.cache_lock:
                self.local_cache[key] = value
                self.cache_version = self.context_version.value

            print(f"Context updated: {key} = {value} (version {self.context_version.value}, process {process_id})")

    def get_context(self, process_id: int = 0) -> dict[str, Any]:
        """Return dict(self.shared_context) with caching for performance"""
        current_version = self.context_version.value

        # Check if local cache is current
        with self.cache_lock:
            if self.cache_version == current_version and self.local_cache:
                print(f"Context retrieved from cache (version {current_version}, process {process_id})")
                return dict(self.local_cache)

        # Cache is stale, update from shared context
        with self.context_lock:
            context_copy = dict(self.shared_context)

            # Update local cache
            with self.cache_lock:
                self.local_cache = context_copy.copy()
                self.cache_version = current_version

            print(f"Context retrieved from shared memory (version {current_version}, process {process_id})")
            return context_copy

    def get_context_value(self, key: str, default: Any = None, process_id: int = 0) -> Any:
        """Get specific context value"""
        context = self.get_context(process_id)
        return context.get(key, default)

    def get_context_version(self) -> int:
        """Get current context version for synchronization"""
        return self.context_version.value

    def sync_check(self, process_id: int = 0) -> bool:
        """Check if local cache is synchronized with shared context"""
        with self.cache_lock:
            return self.cache_version == self.context_version.value

    def force_sync(self, process_id: int = 0):
        """Force synchronization of local cache with shared context"""
        self.get_context(process_id)  # This will update cache if needed

    def clear_context(self):
        """Clear all context data"""
        with self.context_lock:
            self.shared_context.clear()
            self.context_version.value += 1

            with self.cache_lock:
                self.local_cache.clear()
                self.cache_version = self.context_version.value

            print("Context cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get context manager statistics"""
        return {
            "shared_context_size": len(self.shared_context),
            "current_version": self.context_version.value,
            "cache_size": len(self.local_cache),
            "cache_version": self.cache_version,
            "cache_synchronized": self.sync_check(),
        }
