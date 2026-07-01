# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import ModelRequest, TextInput

from ._helpers import start_grpc_pair


@pytest.mark.asyncio
class TestE2EGrpc:
    async def test_single_turn_round_trip_polling(self) -> None:
        pair = await start_grpc_pair("grpc pong", streaming=False)

        try:
            reply = await pair.client.ask("grpc ping")

            assert reply.response.content == "grpc pong"
        finally:
            await pair.grpc_server.stop(grace=0)

    async def test_single_turn_round_trip_streaming(self) -> None:
        pair = await start_grpc_pair("grpc streamed", streaming=True)

        try:
            reply = await pair.client.ask("grpc ping")

            assert reply.response.content == "grpc streamed"
        finally:
            await pair.grpc_server.stop(grace=0)

    async def test_multi_turn_history_propagated_through_grpc(self) -> None:
        pair = await start_grpc_pair("ack", streaming=False)

        try:
            reply1 = await pair.client.ask("first")
            assert reply1.response.content == "ack"

            reply2 = await reply1.ask("second")
            assert reply2.response.content == "ack"

            pair.tracking.mock.assert_called_with(ModelRequest([TextInput("second")]))
        finally:
            await pair.grpc_server.stop(grace=0)
