# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextInput,
)

from ._helpers import make_recording_pair


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_server_sees_full_history_on_second_turn(streaming: bool) -> None:
    pair = make_recording_pair("ok", streaming=streaming)

    reply1 = await pair.client.ask("first user message")
    assert reply1.response.content == "ok"

    reply2 = await reply1.ask("second user message")
    assert reply2.response.content == "ok"

    [first_call, second_call] = pair.recording.calls

    assert first_call == [ModelRequest([TextInput("first user message")])]
    assert second_call == [
        ModelRequest([TextInput("first user message")]),
        ModelResponse(message=ModelMessage("ok")),
        ModelRequest([TextInput("second user message")]),
    ]
