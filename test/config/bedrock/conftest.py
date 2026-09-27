# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator

import pytest_asyncio
from aiohttp.test_utils import TestServer

from test.config.bedrock._helpers import FakeBedrock


@pytest_asyncio.fixture
async def bedrock() -> AsyncGenerator[FakeBedrock]:
    fake = FakeBedrock()
    async with TestServer(fake.app()) as server:
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        fake.release()
