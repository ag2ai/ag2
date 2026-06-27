# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture()
def mock() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def async_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def signal() -> asyncio.Event:
    return asyncio.Event()
