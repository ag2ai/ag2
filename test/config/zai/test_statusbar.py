# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from fast_depends.use import SerializerCls

from ag2 import TextInput
from ag2.config.zai.mappers import convert_messages
from ag2.events import ModelRequest
from test.statusbar_support import status_model_inputs


@pytest.mark.asyncio
async def test_retained_status_parts_use_normal_zai_conversion() -> None:
    messages = await status_model_inputs(TextInput("saved"), persist=True)
    requests = [event for event in messages if isinstance(event, ModelRequest)]

    assert convert_messages([], requests, SerializerCls) == [
        {"role": "user", "content": "start"},
        {"role": "user", "content": "saved"},
        {"role": "user", "content": "saved"},
    ]
