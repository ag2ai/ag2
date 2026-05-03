# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Artifact, Part, Role, TaskArtifactUpdateEvent

from autogen.beta.a2a.mappers import (
    a2a_message_to_inputs,
    a2a_parts_to_inputs,
    artifact_text,
    followup_user_message,
    input_required_message,
    inputs_to_a2a_parts,
    model_request_to_a2a_message,
    task_artifact_update_to_events,
)
from autogen.beta.events import (
    BinaryInput,
    DataInput,
    FileIdInput,
    ModelMessageChunk,
    ModelRequest,
    TextInput,
    UrlInput,
)
from autogen.beta.events.input_events import BinaryType


class TestPartRoundTrip:
    def test_text_input(self) -> None:
        original = TextInput("hello")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_text_input_preserves_user_metadata(self) -> None:
        original = TextInput("hi", metadata={"locale": "en"})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_binary_input_image(self) -> None:
        original = BinaryInput(b"\x00\xff", media_type="image/png", kind=BinaryType.IMAGE)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_binary_input_preserves_vendor_metadata(self) -> None:
        original = BinaryInput(b"data", media_type="image/png", vendor_metadata={"filename": "x.png"})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_file_id_input_with_filename(self) -> None:
        original = FileIdInput("file_123", filename="report.pdf")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_file_id_input_without_filename(self) -> None:
        original = FileIdInput("file_xyz")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_url_input(self) -> None:
        original = UrlInput("https://x.com/a.png", kind=BinaryType.IMAGE)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_data_input_dict(self) -> None:
        original = DataInput({"k": "v", "n": 1})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_data_input_string(self) -> None:
        original = DataInput("simple-string")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original

    def test_data_input_int(self) -> None:
        original = DataInput(42)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert restored == original


class TestModelRequestRoundTrip:
    def test_text_only_request(self) -> None:
        req = ModelRequest([TextInput("hi"), TextInput("there")])

        msg = model_request_to_a2a_message(req, context_id="ctx-1")

        assert a2a_message_to_inputs(msg) == [TextInput("hi"), TextInput("there")]

    def test_role_is_user(self) -> None:
        msg = model_request_to_a2a_message(ModelRequest([TextInput("hi")]), context_id="ctx-1")

        assert msg.role == Role.ROLE_USER

    def test_context_and_task_ids_propagate(self) -> None:
        msg = model_request_to_a2a_message(ModelRequest([TextInput("hi")]), context_id="ctx-1", task_id="task-9")

        assert (msg.context_id, msg.task_id) == ("ctx-1", "task-9")

    def test_mixed_part_types_round_trip(self) -> None:
        req = ModelRequest([TextInput("hi"), DataInput({"k": "v"})])

        msg = model_request_to_a2a_message(req, context_id="ctx-1")

        assert a2a_message_to_inputs(msg) == [TextInput("hi"), DataInput({"k": "v"})]


class TestArtifactStreaming:
    def test_text_part_yields_chunk(self) -> None:
        artifact = Artifact(artifact_id="a-1", parts=[Part(text="abc")])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        assert list(task_artifact_update_to_events(event)) == [ModelMessageChunk("abc")]

    def test_multiple_text_parts_yield_separate_chunks(self) -> None:
        artifact = Artifact(artifact_id="a-1", parts=[Part(text="hello "), Part(text="world")])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        assert list(task_artifact_update_to_events(event)) == [
            ModelMessageChunk("hello "),
            ModelMessageChunk("world"),
        ]

    def test_empty_text_parts_dropped(self) -> None:
        artifact = Artifact(artifact_id="a-1", parts=[Part(text=""), Part(text="kept")])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        assert list(task_artifact_update_to_events(event)) == [ModelMessageChunk("kept")]

    def test_artifact_text_concatenates_text_parts(self) -> None:
        artifact = Artifact(artifact_id="a-1", parts=[Part(text="hello "), Part(text="world")])

        assert artifact_text(artifact) == "hello world"


class TestControlMessages:
    def test_input_required_uses_agent_role(self) -> None:
        msg = input_required_message("Please confirm", context_id="c-1", task_id="t-1")

        assert msg.role == Role.ROLE_AGENT
        assert (msg.context_id, msg.task_id) == ("c-1", "t-1")

    def test_followup_uses_user_role(self) -> None:
        msg = followup_user_message("yes", context_id="c-1", task_id="t-1")

        assert msg.role == Role.ROLE_USER
        assert (msg.context_id, msg.task_id) == ("c-1", "t-1")


def test_user_metadata_with_reserved_prefix_is_dropped() -> None:
    [restored] = a2a_parts_to_inputs(
        inputs_to_a2a_parts([TextInput("hi", metadata={"ag2_beta_kind": "user-supplied", "locale": "en"})])
    )

    assert restored == TextInput("hi", metadata={"locale": "en"})
