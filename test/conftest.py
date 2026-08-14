# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

# The gRPC C-core (pulled in by `test/a2a`) installs pthread_atfork handlers. On macOS it
# runs the poll engine, whose post-fork child handler logs "FD from fork parent still in
# poll list" at INFO to fd 2 — so every later test that spawns a subprocess with stderr
# merged into stdout (LocalSandbox, the ACP terminal) captures that noise and its
# exact-output assertions fail. Silence gRPC's INFO chatter before the core initialises.
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")


@pytest.fixture()
def mock() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def async_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def signal() -> asyncio.Event:
    return asyncio.Event()
