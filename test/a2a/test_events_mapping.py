# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import datetime, timezone

import pytest
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from ag2.a2a.events import (
    A2AMessage,
    A2ATaskArtifactUpdate,
    A2ATaskSnapshot,
    A2ATaskStatusUpdate,
    A2ATextArtifact,
    A2AToolCallArtifact,
)
from ag2.a2a.extension import MIME_TOOL_CALL
from ag2.a2a.mappers.events import (
    a2a_event_to_sdk,
    chunk_to_text_artifact,
    client_call_to_artifact,
    parse_artifact_update,
    parse_stream_response,
    parse_task_artifact,
    task_state_to_status_update,
)
from ag2.a2a.mappers.parts import data_part, struct_to_dict
from ag2.a2a.mappers.tools import call_to_payload
from ag2.events import ClientToolCallEvent, ModelMessageChunk, ToolCallEvent


class TestAg2ToA2A:
    def test_chunk_to_text_artifact(self) -> None:
        ev = chunk_to_text_artifact(
            ModelMessageChunk("hello"),
            artifact_id="art-1",
            task_id="t-1",
            context_id="c-1",
        )

        assert ev == A2ATextArtifact(
            update=TaskArtifactUpdateEvent(
                task_id="t-1",
                context_id="c-1",
                artifact=Artifact(artifact_id="art-1", parts=[Part(text="hello")]),
                append=True,
                last_chunk=False,
            ),
            append=True,
            last_chunk=False,
            text="hello",
        )

    def test_client_call_to_artifact(self) -> None:
        call = ToolCallEvent(id="call-1", name="lookup", arguments='{"q":"x"}')

        ev = client_call_to_artifact(
            ClientToolCallEvent(id="call-1", name="lookup", arguments='{"q":"x"}'),
            task_id="t-1",
            context_id="c-1",
        )

        assert ev == A2AToolCallArtifact(
            update=TaskArtifactUpdateEvent(
                task_id="t-1",
                context_id="c-1",
                artifact=Artifact(
                    artifact_id="call-1",
                    name="lookup",
                    parts=[data_part(call_to_payload(call), media_type=MIME_TOOL_CALL)],
                ),
                append=False,
                last_chunk=True,
            ),
            append=False,
            last_chunk=True,
            call=call,
        )

    def test_task_state_to_status_update(self) -> None:
        ev = task_state_to_status_update(
            TaskState.TASK_STATE_COMPLETED,
            task_id="t-1",
            context_id="c-1",
        )

        assert ev == A2ATaskStatusUpdate(
            update=TaskStatusUpdateEvent(
                task_id="t-1",
                context_id="c-1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            ),
            state=TaskState.TASK_STATE_COMPLETED,
        )


class TestA2AToSdk:
    def test_artifact_update_unwrap(self) -> None:
        ev = chunk_to_text_artifact(
            ModelMessageChunk("hi"),
            artifact_id="a",
            task_id="t",
            context_id="c",
        )

        assert a2a_event_to_sdk(ev) is ev.update

    def test_status_update_unwrap(self) -> None:
        ev = task_state_to_status_update(TaskState.TASK_STATE_FAILED, task_id="t", context_id="c")

        proto = a2a_event_to_sdk(ev)

        assert proto == TaskStatusUpdateEvent(
            task_id="t",
            context_id="c",
            status=TaskStatus(state=TaskState.TASK_STATE_FAILED),
        )

    def test_message_unwrap(self) -> None:
        msg = Message(message_id="m-1", role=Role.ROLE_AGENT, parts=[Part(text="ok")])

        assert a2a_event_to_sdk(A2AMessage(message=msg)) is msg

    def test_task_snapshot_unwrap(self) -> None:
        task = Task(id="t-1", context_id="c-1", status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED))

        assert a2a_event_to_sdk(A2ATaskSnapshot(task=task)) is task


class TestSdkToA2A:
    def test_text_artifact(self) -> None:
        update = TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=Artifact(artifact_id="a", parts=[Part(text="streamed")]),
            append=True,
            last_chunk=False,
        )

        ev = parse_stream_response(StreamResponse(artifact_update=update))

        assert ev == A2ATextArtifact(update=update, append=True, last_chunk=False, text="streamed")

    def test_tool_call_artifact(self) -> None:
        payload = {"id": "c-1", "name": "lookup", "arguments": json.dumps({"q": "x"})}
        update = TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=Artifact(
                artifact_id="c-1",
                parts=[data_part(payload, media_type=MIME_TOOL_CALL)],
            ),
            append=False,
            last_chunk=True,
        )

        ev = parse_stream_response(StreamResponse(artifact_update=update))

        assert ev == A2AToolCallArtifact(
            update=update,
            append=False,
            last_chunk=True,
            call=ToolCallEvent(id="c-1", name="lookup", arguments=json.dumps({"q": "x"})),
        )

    def test_status_update(self) -> None:
        update = TaskStatusUpdateEvent(
            task_id="t",
            context_id="c",
            status=TaskStatus(state=TaskState.TASK_STATE_INPUT_REQUIRED),
        )

        ev = parse_stream_response(StreamResponse(status_update=update))

        assert ev == A2ATaskStatusUpdate(update=update, state=TaskState.TASK_STATE_INPUT_REQUIRED)

    def test_task_snapshot(self) -> None:
        task = Task(id="t", context_id="c", status=TaskStatus(state=TaskState.TASK_STATE_WORKING))

        ev = parse_stream_response(StreamResponse(task=task))

        assert ev == A2ATaskSnapshot(task=task)

    def test_message(self) -> None:
        msg = Message(message_id="m", role=Role.ROLE_AGENT, parts=[Part(text="reply")])

        ev = parse_stream_response(StreamResponse(message=msg))

        assert ev == A2AMessage(message=msg)

    def test_unknown_artifact_falls_back_to_base(self) -> None:
        # Mixed text + opaque data: not text-only, not single tool-call+json — falls
        # through to plain A2ATaskArtifactUpdate so unknown extensions still surface.
        update = TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=Artifact(
                artifact_id="a",
                parts=[
                    Part(text="head"),
                    data_part({"x": 1}, media_type="application/vnd.example+json"),
                ],
            ),
        )

        ev = parse_stream_response(StreamResponse(artifact_update=update))

        assert type(ev) is A2ATaskArtifactUpdate
        assert ev == A2ATaskArtifactUpdate(update=update, append=False, last_chunk=False)


class TestParseTaskArtifact:
    def test_text_artifact_from_polling_snapshot(self) -> None:
        artifact = Artifact(artifact_id="a", parts=[Part(text="final text")])

        ev = parse_task_artifact(artifact, task_id="t", context_id="c")

        # Polling treats every artifact as final, non-appended.
        assert ev == A2ATextArtifact(
            update=TaskArtifactUpdateEvent(
                task_id="t",
                context_id="c",
                artifact=artifact,
                append=False,
                last_chunk=True,
            ),
            append=False,
            last_chunk=True,
            text="final text",
        )

    def test_tool_call_artifact_from_polling_snapshot(self) -> None:
        payload = {"id": "c-1", "name": "f", "arguments": "{}"}
        artifact = Artifact(
            artifact_id="c-1",
            parts=[data_part(payload, media_type=MIME_TOOL_CALL)],
        )

        ev = parse_task_artifact(artifact, task_id="t", context_id="c")

        assert ev == A2AToolCallArtifact(
            update=TaskArtifactUpdateEvent(
                task_id="t",
                context_id="c",
                artifact=artifact,
                append=False,
                last_chunk=True,
            ),
            append=False,
            last_chunk=True,
            call=ToolCallEvent(id="c-1", name="f", arguments="{}"),
        )

    def test_routes_through_same_classifier_as_streaming(self) -> None:
        # Confirms both transport paths converge on the same typed view.
        artifact = Artifact(artifact_id="a", parts=[Part(text="x")])
        polled = parse_task_artifact(artifact, task_id="t", context_id="c")
        streamed = parse_artifact_update(polled.update)

        assert type(polled) is type(streamed)


@pytest.mark.asyncio
class TestRoundTrip:
    async def test_chunk_artifact_roundtrip(self) -> None:
        original = chunk_to_text_artifact(
            ModelMessageChunk("hello"),
            artifact_id="a-1",
            task_id="t-1",
            context_id="c-1",
            append=True,
            last_chunk=True,
        )

        reparsed = parse_stream_response(StreamResponse(artifact_update=a2a_event_to_sdk(original)))

        assert reparsed == original


class TestWireMetadata:
    def test_status_update_timestamp_propagates(self) -> None:
        ts = datetime(2026, 5, 11, 12, 30, 45, tzinfo=timezone.utc)
        ev = task_state_to_status_update(
            TaskState.TASK_STATE_COMPLETED,
            task_id="t",
            context_id="c",
            timestamp=ts,
        )

        assert ev.update.status.timestamp.seconds == int(ts.timestamp())

    def test_status_update_omits_timestamp_by_default(self) -> None:
        ev = task_state_to_status_update(TaskState.TASK_STATE_WORKING, task_id="t", context_id="c")

        assert ev.update.status.timestamp.seconds == 0

    def test_chunk_artifact_optional_name_and_description(self) -> None:
        ev = chunk_to_text_artifact(
            ModelMessageChunk("hi"),
            artifact_id="a",
            task_id="t",
            context_id="c",
            name="streamed-output",
            description="model text stream",
        )

        assert ev.update.artifact.name == "streamed-output"
        assert ev.update.artifact.description == "model text stream"

    def test_chunk_artifact_metadata_round_trips_through_struct(self) -> None:
        ev = chunk_to_text_artifact(
            ModelMessageChunk("hi"),
            artifact_id="a",
            task_id="t",
            context_id="c",
            artifact_metadata={"source": "llm", "chunk_index": 3},
        )

        assert struct_to_dict(ev.update.artifact.metadata) == {"source": "llm", "chunk_index": 3}

    def test_client_call_artifact_uses_tool_name_by_default(self) -> None:
        ev = client_call_to_artifact(
            ClientToolCallEvent(id="c-1", name="search", arguments="{}"),
            task_id="t",
            context_id="c",
        )

        assert ev.update.artifact.name == "search"

    def test_client_call_artifact_name_override(self) -> None:
        ev = client_call_to_artifact(
            ClientToolCallEvent(id="c-1", name="search", arguments="{}"),
            task_id="t",
            context_id="c",
            name="custom-label",
        )

        assert ev.update.artifact.name == "custom-label"
