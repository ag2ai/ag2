# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.acp.session import new_prompt_text
from ag2.events import ModelRequest, TextInput
from ag2.events.types import ModelMessage


def _req(text: str) -> ModelRequest:
    return ModelRequest(parts=[TextInput(text)])


def test_delta_returns_only_new_requests() -> None:
    msgs = [_req("first"), ModelMessage("reply"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=0)
    assert "first" in text and "second" in text
    assert count == 2


def test_delta_skips_already_sent() -> None:
    msgs = [_req("first"), _req("second")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == "second"
    assert count == 2


def test_delta_empty_when_nothing_new() -> None:
    msgs = [_req("only")]
    text, count = new_prompt_text(msgs, sent_count=1)
    assert text == ""
    assert count == 1
