# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.config.gemini.events import GeminiToolCallEvent
from ag2.events._serialization import deserialize_value, serialize_value


def test_thought_signature_round_trip() -> None:
    sig = b"\x12\x34\n\x32\x01\x0c\x39\xd6\xc7\xa3"
    event = GeminiToolCallEvent(
        id="call-1",
        name="get_weather",
        arguments='{"city": "Paris"}',
        thought_signature=sig,
    )

    round_tripped = deserialize_value(serialize_value(event))

    assert isinstance(round_tripped, GeminiToolCallEvent)
    assert round_tripped.thought_signature == sig
    assert isinstance(round_tripped.thought_signature, bytes)


def test_thought_signature_defaults_to_none() -> None:
    event = GeminiToolCallEvent(id="call-1", name="get_weather", arguments="{}")

    assert event.thought_signature is None
