# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent, MemoryStream
from autogen.beta.events import ModelReasoning
from autogen.beta.testing import TestConfig


@pytest.mark.asyncio
async def test_client_receives_reasoning_event_from_server(serve) -> None:
    server_agent = Agent(
        "thinker",
        "p",
        config=TestConfig(ModelReasoning("considering..."), "the answer is 42"),
    )
    config = serve(server_agent)

    seen: list[str] = []
    stream = MemoryStream()

    @stream.where(ModelReasoning).subscribe
    def record(event: ModelReasoning) -> None:
        seen.append(event.content)

    reply = await Agent("client", "p", config=config).ask("hi", stream=stream)

    assert reply.body == "the answer is 42"
    assert seen == ["considering..."]
