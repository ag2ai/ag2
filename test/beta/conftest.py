# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest


@pytest.fixture()
def async_mock() -> AsyncMock:
    return AsyncMock()


@pytest.fixture()
def signal() -> asyncio.Event:
    return asyncio.Event()
