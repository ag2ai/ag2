# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import Agent
from autogen.beta.a2a.mappers import (
    HISTORY_KEY,
    message_metadata,
    model_request_to_a2a_message,
)
from autogen.beta.events import (
    HumanMessage,
    ModelRequest,
    TextInput,
)
from autogen.beta.testing import TestConfig, TrackingConfig


@pytest.mark.asyncio
async def test_second_turn_seeds_server_with_first_turn_events(serve) -> None:
    tracking = TrackingConfig(TestConfig("ok"))
    config = serve(Agent("specialist", "p", config=tracking))

    client_agent = Agent("client", "p", config=config)
    reply1 = await client_agent.ask("first question")
    reply2 = await reply1.ask("second question")

    assert reply1.body == "ok"
    assert reply2.body == "ok"

    first_call, second_call = tracking.calls
    assert first_call == [ModelRequest([TextInput("first question")])]
    assert ModelRequest([TextInput("first question")]) in second_call
    assert second_call[-1] == ModelRequest([TextInput("second question")])


def test_user_metadata_coexists_with_history_on_wire() -> None:
    msg = model_request_to_a2a_message(
        ModelRequest([TextInput("hi")]),
        context_id="ctx",
        metadata={"my_app_key": 7},
        history=(HumanMessage("prior"),),
    )

    md = message_metadata(msg)

    assert md == IsPartialDict({"my_app_key": 7})
    assert HISTORY_KEY in md
