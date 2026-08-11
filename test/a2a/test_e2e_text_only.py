# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ModelRequest, TextInput

from ._helpers import make_pair


@pytest.mark.asyncio
class TestE2ETextOnly:
    async def test_single_turn_round_trip(self) -> None:
        pair = make_pair("hello world", streaming=False)

        reply = await pair.client.ask("ping")

        assert reply.response.content == "hello world"

    async def test_streaming_round_trip(self) -> None:
        pair = make_pair("streamed", streaming=True)

        reply = await pair.client.ask("ping")

        assert reply.response.content == "streamed"

    async def test_server_sees_user_input_in_history(self) -> None:
        pair = make_pair("ack", streaming=False)

        await pair.client.ask("hello server")

        pair.tracking.mock.assert_called_with(ModelRequest([TextInput("hello server")]))
