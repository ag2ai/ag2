# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from base64 import b64encode
from uuid import uuid4

from a2a.types import (
    Artifact,
    DataPart,
    FilePart,
    FileWithBytes,
    FileWithUri,
    Message,
    Part,
    Role,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TextPart,
)

from autogen.beta.a2a.mappers import (
    a2a_message_to_inputs,
    a2a_parts_to_inputs,
    artifact_text,
    followup_user_message,
    input_required_message,
    inputs_to_a2a_parts,
    model_request_to_a2a_message,
    task_artifact_update_to_chunks,
    task_history_to_events,
)
from autogen.beta.events import (
    BinaryInput,
    DataInput,
    ModelRequest,
    ModelResponse,
    TextInput,
)
from autogen.beta.events.input_events import BinaryType, FileIdInput, UrlInput


class TestTextRoundTrip:
    def test_text_input_to_text_part(self) -> None:
        [part] = inputs_to_a2a_parts([TextInput("hello")])

        assert isinstance(part.root, TextPart)
        assert part.root.text == "hello"

    def test_text_part_to_text_input(self) -> None:
        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([TextInput("hello")]))

        assert isinstance(restored, TextInput)
        assert restored.content == "hello"

    def test_user_metadata_is_preserved(self) -> None:
        original = TextInput("hi", metadata={"locale": "en"})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, TextInput)
        assert restored.metadata == {"locale": "en"}


class TestBinaryRoundTrip:
    def test_binary_input_to_file_part_with_bytes(self) -> None:
        [part] = inputs_to_a2a_parts([BinaryInput(b"\x00\xff", media_type="image/png", kind=BinaryType.IMAGE)])

        assert isinstance(part.root, FilePart)
        assert isinstance(part.root.file, FileWithBytes)
        assert part.root.file.bytes == b64encode(b"\x00\xff").decode("ascii")
        assert part.root.file.mime_type == "image/png"

    def test_round_trip_preserves_data_media_type_and_kind(self) -> None:
        original = BinaryInput(b"raw-bytes", media_type="audio/wav", kind=BinaryType.AUDIO)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, BinaryInput)
        assert restored.data == b"raw-bytes"
        assert restored.media_type == "audio/wav"
        assert restored.kind == BinaryType.AUDIO

    def test_vendor_metadata_is_preserved(self) -> None:
        original = BinaryInput(b"data", media_type="image/png", vendor_metadata={"filename": "x.png"})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, BinaryInput)
        assert restored.vendor_metadata == {"filename": "x.png"}


class TestFileIdRoundTrip:
    def test_file_id_input_to_data_part(self) -> None:
        [part] = inputs_to_a2a_parts([FileIdInput("file_123", filename="report.pdf")])

        assert isinstance(part.root, DataPart)
        assert part.root.data == {"file_id": "file_123", "filename": "report.pdf"}

    def test_round_trip_preserves_id_and_filename(self) -> None:
        original = FileIdInput("file_123", filename="report.pdf")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, FileIdInput)
        assert restored.file_id == "file_123"
        assert restored.filename == "report.pdf"

    def test_round_trip_without_filename(self) -> None:
        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([FileIdInput("file_xyz")]))

        assert isinstance(restored, FileIdInput)
        assert restored.file_id == "file_xyz"
        assert restored.filename is None


class TestUrlRoundTrip:
    def test_url_input_to_file_part_with_uri(self) -> None:
        [part] = inputs_to_a2a_parts([UrlInput("https://x.com/a.png", kind=BinaryType.IMAGE)])

        assert isinstance(part.root, FilePart)
        assert isinstance(part.root.file, FileWithUri)
        assert part.root.file.uri == "https://x.com/a.png"

    def test_round_trip_preserves_url_and_kind(self) -> None:
        original = UrlInput("https://x.com/a.png", kind=BinaryType.IMAGE)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, UrlInput)
        assert restored.url == "https://x.com/a.png"
        assert restored.kind == BinaryType.IMAGE


class TestDataInputRoundTrip:
    def test_dict_data_round_trip(self) -> None:
        original = DataInput({"k": "v", "n": 1})

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, DataInput)
        assert restored.data == {"k": "v", "n": 1}

    def test_string_data_round_trip(self) -> None:
        original = DataInput("simple-string")

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, DataInput)
        assert restored.data == "simple-string"

    def test_int_data_round_trip(self) -> None:
        original = DataInput(42)

        [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

        assert isinstance(restored, DataInput)
        assert restored.data == 42


class TestModelRequestRoundTrip:
    def test_model_request_to_message(self) -> None:
        req = ModelRequest([TextInput("hi"), DataInput({"k": "v"})])

        msg = model_request_to_a2a_message(req, context_id="ctx-1", task_id="task-9")

        assert msg.role == Role.user
        assert msg.context_id == "ctx-1"
        assert msg.task_id == "task-9"
        assert len(msg.parts) == 2

    def test_message_to_inputs(self) -> None:
        req = ModelRequest([TextInput("hi"), TextInput("there")])
        msg = model_request_to_a2a_message(req, context_id=None)

        restored = a2a_message_to_inputs(msg)

        assert [type(r).__name__ for r in restored] == ["TextInput", "TextInput"]
        assert [r.content for r in restored if isinstance(r, TextInput)] == ["hi", "there"]


class TestArtifactStreaming:
    def test_text_part_yields_chunk(self) -> None:
        artifact = Artifact(artifact_id=uuid4().hex, parts=[Part(root=TextPart(text="abc"))])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        chunks = list(task_artifact_update_to_chunks(event))

        assert [c.content for c in chunks] == ["abc"]

    def test_data_part_with_content_yields_chunk(self) -> None:
        artifact = Artifact(artifact_id=uuid4().hex, parts=[Part(root=DataPart(data={"content": "x"}))])
        event = TaskArtifactUpdateEvent(
            task_id="t-1", context_id="c-1", artifact=artifact, append=True, last_chunk=False
        )

        chunks = list(task_artifact_update_to_chunks(event))

        assert [c.content for c in chunks] == ["x"]

    def test_artifact_text_concatenates_text_parts(self) -> None:
        artifact = Artifact(
            artifact_id=uuid4().hex,
            parts=[
                Part(root=TextPart(text="hello ")),
                Part(root=TextPart(text="world")),
            ],
        )

        assert artifact_text(artifact) == "hello world"


class TestTaskHistory:
    def test_user_message_becomes_model_request(self) -> None:
        msg = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="hi"))],
            message_id=uuid4().hex,
            context_id="c-1",
        )
        task = Task(
            id="t-1",
            context_id="c-1",
            status=TaskStatus(state=TaskState.completed),
            history=[msg],
        )

        events = task_history_to_events(task)

        assert len(events) == 1
        assert isinstance(events[0], ModelRequest)
        assert isinstance(events[0].parts[0], TextInput)
        assert events[0].parts[0].content == "hi"  # type: ignore[union-attr]

    def test_agent_message_becomes_model_response(self) -> None:
        msg = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text="ok"))],
            message_id=uuid4().hex,
            context_id="c-1",
        )
        task = Task(
            id="t-1",
            context_id="c-1",
            status=TaskStatus(state=TaskState.completed),
            history=[msg],
        )

        events = task_history_to_events(task)

        assert len(events) == 1
        assert isinstance(events[0], ModelResponse)
        assert events[0].message is not None
        assert events[0].message.content == "ok"

    def test_empty_history_yields_no_events(self) -> None:
        task = Task(
            id="t-1",
            context_id="c-1",
            status=TaskStatus(state=TaskState.completed),
            history=[],
        )

        assert task_history_to_events(task) == []


class TestControlMessages:
    def test_input_required_uses_agent_role(self) -> None:
        msg = input_required_message("Please confirm", context_id="c-1", task_id="t-1")

        assert msg.role == Role.agent
        assert msg.context_id == "c-1"
        assert msg.task_id == "t-1"
        assert isinstance(msg.parts[0].root, TextPart)
        assert msg.parts[0].root.text == "Please confirm"

    def test_followup_uses_user_role(self) -> None:
        msg = followup_user_message("yes", context_id="c-1", task_id="t-1")

        assert msg.role == Role.user
        assert msg.context_id == "c-1"
        assert msg.task_id == "t-1"
        assert isinstance(msg.parts[0].root, TextPart)
        assert msg.parts[0].root.text == "yes"


def test_user_metadata_keys_with_reserved_prefix_are_dropped() -> None:
    # The `ag2_beta_*` prefix is reserved for the mapper's internal markers.
    # User metadata using that prefix is silently dropped on round-trip —
    # documented contract; non-prefixed keys survive untouched.
    user_meta = {"ag2_beta_kind": "user-supplied", "locale": "en"}
    original = TextInput("hi", metadata=user_meta)

    [restored] = a2a_parts_to_inputs(inputs_to_a2a_parts([original]))

    assert isinstance(restored, TextInput)
    assert restored.metadata == {"locale": "en"}
