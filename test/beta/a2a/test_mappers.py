# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64encode
from uuid import uuid4

from a2a.types import (
    Artifact,
    DataPart,
    Part,
    Role,
    TaskArtifactUpdateEvent,
    TextPart,
)
from dirty_equals import IsPartialDict

from autogen.beta.a2a.mappers import (
    a2a_message_to_inputs,
    a2a_parts_to_inputs,
    artifact_text,
    followup_user_message,
    input_required_message,
    inputs_to_a2a_parts,
    model_request_to_a2a_message,
    task_artifact_update_to_chunks,
)
from autogen.beta.events import (
    BinaryInput,
    DataInput,
    ModelMessageChunk,
    ModelRequest,
    TextInput,
)
from autogen.beta.events.input_events import BinaryType, FileIdInput, UrlInput


class TestTextRoundTrip:
    def test_text_input_to_text_part(self) -> None:
        [part] = inputs_to_a2a_parts([TextInput("hello")])

        assert part.model_dump(by_alias=False) == IsPartialDict({
            "kind": "text",
            "text": "hello",
            "metadata": {"ag2_beta_kind": "text"},
        })

    def test_round_trip_preserves_content(self) -> None:
        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([TextInput("hello")]))

        assert restored == TextInput("hello")

    def test_user_metadata_is_preserved(self) -> None:
        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([TextInput("hi", metadata={"locale": "en"})]))

        assert restored == TextInput("hi", metadata={"locale": "en"})


class TestBinaryRoundTrip:
    def test_binary_input_to_file_part_with_bytes(self) -> None:
        [part] = inputs_to_a2a_parts([BinaryInput(b"\x00\xff", media_type="image/png", kind=BinaryType.IMAGE)])

        assert part.model_dump(by_alias=False) == IsPartialDict({
            "kind": "file",
            "file": IsPartialDict({"bytes": b64encode(b"\x00\xff").decode("ascii"), "mime_type": "image/png"}),
            "metadata": IsPartialDict({"ag2_beta_kind": "binary", "ag2_beta_binary_type": "image"}),
        })

    def test_round_trip_preserves_data_media_type_and_kind(self) -> None:
        original = BinaryInput(b"raw-bytes", media_type="audio/wav", kind=BinaryType.AUDIO)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_vendor_metadata_is_preserved(self) -> None:
        original = BinaryInput(b"data", media_type="image/png", vendor_metadata={"filename": "x.png"})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original


class TestFileIdRoundTrip:
    def test_file_id_input_to_data_part(self) -> None:
        [part] = inputs_to_a2a_parts([FileIdInput("file_123", filename="report.pdf")])

        assert part.model_dump(by_alias=False) == IsPartialDict({
            "kind": "data",
            "data": {"file_id": "file_123", "filename": "report.pdf"},
            "metadata": IsPartialDict({"ag2_beta_kind": "file_id", "ag2_beta_filename": "report.pdf"}),
        })

    def test_round_trip_preserves_id_and_filename(self) -> None:
        original = FileIdInput("file_123", filename="report.pdf")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_round_trip_without_filename(self) -> None:
        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([FileIdInput("file_xyz")]))

        assert restored == FileIdInput("file_xyz")


class TestUrlRoundTrip:
    def test_url_input_to_file_part_with_uri(self) -> None:
        [part] = inputs_to_a2a_parts([UrlInput("https://x.com/a.png", kind=BinaryType.IMAGE)])

        assert part.model_dump(by_alias=False) == IsPartialDict({
            "kind": "file",
            "file": IsPartialDict({"uri": "https://x.com/a.png"}),
            "metadata": IsPartialDict({"ag2_beta_kind": "url", "ag2_beta_binary_type": "image"}),
        })

    def test_round_trip_preserves_url_and_kind(self) -> None:
        original = UrlInput("https://x.com/a.png", kind=BinaryType.IMAGE)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original


class TestDataInputRoundTrip:
    def test_dict_data_round_trip(self) -> None:
        original = DataInput({"k": "v", "n": 1})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_string_data_round_trip(self) -> None:
        original = DataInput("simple-string")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_int_data_round_trip(self) -> None:
        original = DataInput(42)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original


class TestModelRequestRoundTrip:
    def test_model_request_to_message(self) -> None:
        req = ModelRequest([TextInput("hi"), DataInput({"k": "v"})])

        msg = model_request_to_a2a_message(req, context_id="ctx-1", task_id="task-9")

        assert msg.model_dump(by_alias=False) == IsPartialDict({
            "role": Role.user,
            "context_id": "ctx-1",
            "task_id": "task-9",
            "parts": [
                IsPartialDict({"kind": "text", "text": "hi"}),
                IsPartialDict({"kind": "data", "data": {"k": "v"}}),
            ],
        })

    def test_message_to_inputs(self) -> None:
        req = ModelRequest([TextInput("hi"), TextInput("there")])
        msg = model_request_to_a2a_message(req, context_id=None)

        assert a2a_message_to_inputs(msg) == [TextInput("hi"), TextInput("there")]


class TestArtifactStreaming:
    def test_text_part_yields_chunk(self) -> None:
        artifact = Artifact(artifact_id=uuid4().hex, parts=[Part(root=TextPart(text="abc"))])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        assert list(task_artifact_update_to_chunks(event)) == [ModelMessageChunk("abc")]

    def test_data_part_with_content_yields_chunk(self) -> None:
        artifact = Artifact(artifact_id=uuid4().hex, parts=[Part(root=DataPart(data={"content": "x"}))])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        assert list(task_artifact_update_to_chunks(event)) == [ModelMessageChunk("x")]

    def test_artifact_text_concatenates_text_parts(self) -> None:
        artifact = Artifact(
            artifact_id=uuid4().hex,
            parts=[Part(root=TextPart(text="hello ")), Part(root=TextPart(text="world"))],
        )

        assert artifact_text(artifact) == "hello world"


class TestControlMessages:
    def test_input_required_uses_agent_role(self) -> None:
        msg = input_required_message("Please confirm", context_id="c-1", task_id="t-1")

        assert msg.model_dump(by_alias=False) == IsPartialDict({
            "role": Role.agent,
            "context_id": "c-1",
            "task_id": "t-1",
            "parts": [IsPartialDict({"kind": "text", "text": "Please confirm"})],
        })

    def test_followup_uses_user_role(self) -> None:
        msg = followup_user_message("yes", context_id="c-1", task_id="t-1")

        assert msg.model_dump(by_alias=False) == IsPartialDict({
            "role": Role.user,
            "context_id": "c-1",
            "task_id": "t-1",
            "parts": [IsPartialDict({"kind": "text", "text": "yes"})],
        })


def test_user_metadata_keys_with_reserved_prefix_are_dropped() -> None:
    # The `ag2_beta_*` prefix is reserved for the mapper's internal markers.
    # User metadata using that prefix is silently dropped on round-trip —
    # documented contract; non-prefixed keys survive untouched.
    [restored] = a2a_parts_to_inputs(
        inputs_to_a2a_parts([TextInput("hi", metadata={"ag2_beta_kind": "user-supplied", "locale": "en"})])
    )

    assert restored == TextInput("hi", metadata={"locale": "en"})
