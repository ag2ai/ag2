# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``response_id`` is tracing information: absent unless a provider supplied one,
and invisible to equality, like ``model`` and ``provider`` beside it."""

import pytest

from ag2 import Agent, MemoryStream, testing
from ag2.events import ModelMessage, ModelResponse


def test_a_response_has_no_identifier_by_default() -> None:
    assert ModelResponse(ModelMessage("ok")).response_id is None


def test_two_responses_differing_only_in_identifier_compare_equal() -> None:
    assert ModelResponse(ModelMessage("ok"), response_id="resp_1") == ModelResponse(
        ModelMessage("ok"), response_id="resp_2"
    )


@pytest.mark.asyncio
async def test_a_provider_that_supplies_no_identifier_gets_no_placeholder() -> None:
    stream = MemoryStream()
    agent = Agent("a", config=testing.TestConfig("done"))

    await agent.ask("hi", stream=stream)

    [response] = [e for e in await stream.history.get_events() if isinstance(e, ModelResponse)]
    assert response.response_id is None
