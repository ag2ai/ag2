# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Phase 1 network tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from autogen.beta.knowledge import DiskKnowledgeStore, MemoryKnowledgeStore
from autogen.beta.network import ActorIdentity, Hub, Rule


@pytest.fixture
def mem_store() -> MemoryKnowledgeStore:
    return MemoryKnowledgeStore()


@pytest.fixture
def disk_store(tmp_path: Path) -> DiskKnowledgeStore:
    return DiskKnowledgeStore(str(tmp_path / "store"))


@pytest.fixture
def hub(mem_store: MemoryKnowledgeStore) -> Hub:
    return Hub(mem_store)


def make_identity(name: str, **kwargs: Any) -> ActorIdentity:
    return ActorIdentity(name=name, **kwargs)


@pytest.fixture
def identity_factory() -> Any:
    return make_identity
