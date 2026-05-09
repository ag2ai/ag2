# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ._helpers import PromptThenAckExecutor, make_executor_pair


@pytest.mark.asyncio
async def test_input_required_round_trip() -> None:
    executor = PromptThenAckExecutor(prompt="What's your name?")

    async def hitl_hook() -> str:
        return "Semen"

    pair = make_executor_pair(executor, streaming=False, hitl_hook=hitl_hook)

    reply = await pair.client.ask("hello")

    assert reply.response.content == "echo: Semen"
    assert executor.received_user_text == "Semen"
