# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""An adapter may produce a string, an ``Input`` or a sequence of them.

They are converted with the framework's own ``Input.ensure_input``; views then
keep every part, whatever its kind.
"""

import pytest

from ag2.events import DataInput, ImageInput, ModelMessage, ModelRequest, TextInput
from ag2.network.adapters.base import ensure_parts
from ag2.network.views.builtin import FullTranscript, NamedTranscript

from ._helpers import text_envelope, two_party_metadata

_PHOTO = ImageInput("https://example.com/photo.png")


class TestEnsureParts:
    def test_nothing_is_no_parts(self) -> None:
        assert ensure_parts(None) == []

    def test_a_string_is_one_text_part(self) -> None:
        assert ensure_parts("hi") == [TextInput("hi")]

    def test_an_input_is_kept_as_it_is(self) -> None:
        assert ensure_parts(_PHOTO) == [_PHOTO]

    def test_structured_data_is_one_data_part(self) -> None:
        assert ensure_parts({"a": 1}) == [DataInput({"a": 1})]

    @pytest.mark.parametrize("raw", [b"hi", bytearray(b"hi")])
    def test_bytes_are_one_data_part(self, raw: bytes | bytearray) -> None:
        assert ensure_parts(raw) == [DataInput(raw)]

    def test_a_sequence_is_one_part_per_item(self) -> None:
        assert ensure_parts(["hi", _PHOTO, {"a": 1}]) == [TextInput("hi"), _PHOTO, DataInput({"a": 1})]


@pytest.mark.asyncio
class TestViewsWithNonTextParts:
    async def test_a_named_view_labels_a_leading_image_on_its_own(self) -> None:
        projection = await NamedTranscript().project(
            [text_envelope("alice", "bob", "ignored")],
            participant_id="bob",
            channel=two_party_metadata("alice", "bob"),
            render_envelope=lambda envelope: [_PHOTO, TextInput("what is this?")],  # noqa: ARG005
            name_for=lambda agent_id: agent_id.title(),
        )

        assert projection == [ModelRequest([TextInput("[Alice]:"), _PHOTO, TextInput("what is this?")])]

    async def test_an_own_envelope_projects_only_its_text(self) -> None:
        projection = await FullTranscript().project(
            [text_envelope("bob", "alice", "ignored")],
            participant_id="bob",
            channel=two_party_metadata("alice", "bob"),
            render_envelope=lambda envelope: [TextInput("here"), _PHOTO],  # noqa: ARG005
        )

        assert projection == [ModelMessage("here")]
