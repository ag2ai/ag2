# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.a2a.extension import CONTEXT_UPDATE_METADATA_KEY
from ag2.a2a.mappers.messages import build_user_message, extract_context_update
from ag2.a2a.mappers.parts import struct_to_dict
from ag2.events import TextInput


class TestExtraMetadata:
    def test_extra_metadata_merged_alongside_context_update(self) -> None:
        msg = build_user_message(
            [TextInput("hi")],
            context_update={"k": "v"},
            extra_metadata={"trace_id": "abc"},
        )

        assert extract_context_update(msg) == {"k": "v"}
        assert struct_to_dict(msg.metadata)["trace_id"] == "abc"

    def test_extra_metadata_alone(self) -> None:
        msg = build_user_message([TextInput("hi")], extra_metadata={"trace_id": "abc"})

        assert struct_to_dict(msg.metadata) == {"trace_id": "abc"}
        assert extract_context_update(msg) == {}

    def test_ag2_keys_win_on_collision(self) -> None:
        msg = build_user_message(
            [TextInput("hi")],
            context_update={"real": True},
            extra_metadata={CONTEXT_UPDATE_METADATA_KEY: {"fake": True}},
        )

        assert extract_context_update(msg) == {"real": True}

    def test_no_metadata_when_both_empty(self) -> None:
        msg = build_user_message([TextInput("hi")])

        assert not msg.HasField("metadata")
