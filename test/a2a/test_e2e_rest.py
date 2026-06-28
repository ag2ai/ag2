# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ModelRequest, TextInput

from ._helpers import make_rest_pair


@pytest.mark.asyncio
class TestE2ERest:
    async def test_single_turn_round_trip_polling(self) -> None:
        pair = make_rest_pair("rest pong", streaming=False)

        reply = await pair.client.ask("rest ping")

        assert reply.response.content == "rest pong"

    async def test_single_turn_round_trip_streaming(self) -> None:
        pair = make_rest_pair("rest streamed", streaming=True)

        reply = await pair.client.ask("rest ping")

        assert reply.response.content == "rest streamed"

    async def test_multi_turn_history_propagated_through_rest(self) -> None:
        pair = make_rest_pair("ack", streaming=False)

        reply1 = await pair.client.ask("first")
        assert reply1.response.content == "ack"

        reply2 = await reply1.ask("second")
        assert reply2.response.content == "ack"

        pair.tracking.mock.assert_called_with(ModelRequest([TextInput("second")]))
