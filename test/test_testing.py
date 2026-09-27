# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2 import Agent
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestSharedScript:
    async def test_each_client_restarts_the_script_by_default(self) -> None:
        config = TestConfig("first", "second")

        replies = [await Agent("a", config=config).ask("hi") for _ in range(2)]

        assert [r.body for r in replies] == ["first", "first"]

    async def test_a_shared_script_carries_on_across_clients(self) -> None:
        config = TestConfig("first", "second", shared_script=True)

        replies = [await Agent("a", config=config).ask("hi") for _ in range(2)]

        assert [r.body for r in replies] == ["first", "second"]

    async def test_a_shared_script_raises_once_it_runs_out(self) -> None:
        config = TestConfig("only", shared_script=True)
        agent = Agent("a", config=config)
        await agent.ask("hi")

        with pytest.raises(RuntimeError, match="exhausted"):
            await agent.ask("again")
