# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.history import MemoryStorage
from ag2.mcp.server import _build_session_store
from ag2.mcp.sessions import SessionConfig, SessionStore


class _SizedMemoryStorage(MemoryStorage):
    """A storage backend that is falsy while empty because it defines ``__len__``."""

    def __len__(self) -> int:
        return 0


def test_explicit_empty_storage_is_not_replaced() -> None:
    storage = _SizedMemoryStorage()
    assert not storage

    store = SessionStore(storage=storage)

    assert store._storage is storage


def test_session_config_empty_storage_is_not_replaced() -> None:
    storage = _SizedMemoryStorage()

    store = _build_session_store(SessionConfig(storage=storage))

    assert store is not None
    assert store._storage is storage
