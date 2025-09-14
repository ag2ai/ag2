# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp
import threading
from typing import TypeVar

T = TypeVar("T")

__all__ = ["ProcessSafeLock", "ThreadSafeDict"]


class ThreadSafeDict(dict[str, T]):
    """Thread-safe dictionary with locking for concurrent access"""

    def __init__(self):
        super().__init__()
        self._lock = threading.RLock()

    def __getitem__(self, key: str) -> T:
        """Thread-safe get"""
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key: str, value: T):
        """Thread-safe set"""
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: str):
        """Thread-safe delete"""
        with self._lock:
            super().__delitem__(key)

    def get(self, key: str, default: T = None) -> T:
        """Thread-safe get with default"""
        with self._lock:
            return super().get(key, default)

    def pop(self, key: str, default: T = None) -> T:
        """Thread-safe pop"""
        with self._lock:
            return super().pop(key, default)

    def keys(self):
        """Thread-safe keys"""
        with self._lock:
            return list(super().keys())

    def values(self):
        """Thread-safe values"""
        with self._lock:
            return list(super().values())

    def items(self):
        """Thread-safe items"""
        with self._lock:
            return list(super().items())

    def safe_update(self, updates: dict[str, T]):
        """Thread-safe bulk update"""
        with self._lock:
            for key, value in updates.items():
                self[key] = value

    def clear(self):
        """Thread-safe clear"""
        with self._lock:
            super().clear()

    def __len__(self):
        """Thread-safe length"""
        with self._lock:
            return super().__len__()

    def __contains__(self, key: str):
        """Thread-safe contains check"""
        with self._lock:
            return super().__contains__(key)


class ProcessSafeLock:
    """Process-safe lock using multiprocessing for cross-process synchronization"""

    def __init__(self, name: str):
        self.name = name
        self.lock = mp.Lock()

    def __enter__(self):
        """Context manager entry - acquire lock"""
        self.lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release lock"""
        self.lock.release()

    def acquire(self, blocking: bool = True, timeout: float = None) -> bool:
        """Manually acquire the lock"""
        if timeout is not None:
            return self.lock.acquire(blocking, timeout)
        else:
            return self.lock.acquire(blocking)

    def release(self):
        """Manually release the lock"""
        self.lock.release()

    def locked(self) -> bool:
        """Check if lock is currently held"""
        # Note: This is not atomic and should only be used for debugging
        try:
            acquired = self.lock.acquire(False)
            if acquired:
                self.lock.release()
                return False
            return True
        except Exception:
            return True
