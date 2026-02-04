# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest
from a2a.types import Artifact, DataPart, Message, Part, Role, Task, TaskState, TaskStatus, TextPart

from autogen.a2a.utils import (
    CLIENT_TOOLS_KEY,
    CONTEXT_KEY,
    message_from_part,
    message_to_part,
    request_message_from_a2a,
    request_message_to_a2a,
    response_message_from_a2a_artifacts,
    response_message_from_a2a_message,
    response_message_from_a2a_task,
    response_message_to_a2a,
)
from autogen.remote.protocol import RequestMessage, ResponseMessage


class TestMessageToPart:
    def test_simple_text_message(self) -> None:
        message = {"content": "Hello, world!"}
        part = message_to_part(message).root
        assert part == TextPart(text="Hello, world!")

    def test_message_with_metadata(self) -> None:
        message = {"content": "Test content", "role": "assistant", "name": "agent1"}
        part = message_to_part(message).root
        assert part == TextPart(
            text="Test content",
            metadata={"role": "assistant", "name": "agent1"},
        )

    @pytest.mark.parametrize(
        "message",
        (
            pytest.param({"content": "", "role": "user", "name": "test"}, id="empty content"),
            pytest.param({"content": None, "role": "user", "name": "test"}, id="none content"),
            pytest.param({"role": "user", "name": "test"}, id="missing content"),
        ),
    )
    def test_message_to_part_without_content(self, message) -> None:
        part = message_to_part(message).root
        assert part == TextPart(
            text="",
            metadata={"role": "user", "name": "test"},
        )


class TestMessageFromPart:
    def test_text_part_without_metadata(self) -> None:
        part = Part(root=TextPart(text="Test content"))
        message = message_from_part(part)

        assert message == {"content": "Test content"}

    def test_text_part_to_message(self) -> None:
        part = Part(root=TextPart(text="Hello, world!", metadata={"role": "assistant"}))
        message = message_from_part(part)

        assert message == {"role": "assistant", "content": "Hello, world!"}

    def test_data_part_to_message(self) -> None:
        data = {"key": "value", "nested": {"data": "test"}}

        part = Part(root=DataPart(data=data))
        message = message_from_part(part)

        assert message == data

    def test_data_part_with_pydantic_ai_result(self) -> None:
        result_data = {"result": {"answer": "42"}}

        part = Part(root=DataPart(data=result_data, metadata={"json_schema": None}))
        message = message_from_part(part)

        assert message == {"answer": "42"}


class TestRequestMessageToA2A:
    def test_request_with_multiple_messages(self) -> None:
        request = RequestMessage(
            messages=[
                {"content": "Message 1", "role": "user"},
                {"content": "Message 2", "role": "assistant"},
                {"content": "Message 3", "role": "user"},
            ],
            context={"key": "value"},
            client_tools=[{"function": {"name": "tool1"}}],
        )
        context_id = str(uuid4())

        a2a_message = request_message_to_a2a(request, context_id)

        assert a2a_message == Message(
            role=Role.user,
            parts=[
                Part(root=TextPart(text="Message 1", metadata={"role": "user"})),
                Part(root=TextPart(text="Message 2", metadata={"role": "assistant"})),
                Part(root=TextPart(text="Message 3", metadata={"role": "user"})),
            ],
            context_id=context_id,
            metadata={
                CONTEXT_KEY: {"key": "value"},
                CLIENT_TOOLS_KEY: [{"function": {"name": "tool1"}}],
            },
            # randomly generated message_id
            message_id=a2a_message.message_id,
        )


class TestRequestMessageFromA2A:
    def test_simple_a2a_to_request(self) -> None:
        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Hello"))],
            message_id=str(uuid4()),
        )

        request = request_message_from_a2a(a2a_message)

        assert request == RequestMessage(messages=[{"content": "Hello"}])

    def test_a2a_complete_message(self) -> None:
        a2a_message = Message(
            role=Role.user,
            parts=[Part(root=TextPart(text="Test"))],
            message_id=str(uuid4()),
            metadata={
                CONTEXT_KEY: {"key": "value"},
                CLIENT_TOOLS_KEY: [{"function": {"name": "tool1"}}],
            },
        )

        request = request_message_from_a2a(a2a_message)

        assert request == RequestMessage(
            messages=[{"content": "Test"}],
            context={"key": "value"},
            client_tools=[{"function": {"name": "tool1"}}],
        )


class TestResponseMessageFromA2AArtifacts:
    @pytest.mark.parametrize(
        "artifacts",
        (
            pytest.param(None, id="None"),
            pytest.param([], id="Empty list"),
            pytest.param([Artifact(artifact_id=str(uuid4()), name="test", parts=[])], id="Artifact with no parts"),
        ),
    )
    def test_response_message_from_a2a_none_cases(self, artifacts: list[Artifact] | None) -> None:
        result = response_message_from_a2a_artifacts(artifacts)
        assert result is None

    def test_single_artifact_single_part(self) -> None:
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="result",
            parts=[Part(root=TextPart(text="Response text"))],
        )
        result = response_message_from_a2a_artifacts([artifact])

        assert result == ResponseMessage(
            messages=[{"content": "Response text"}],
        )

    def test_artifact_with_context(self) -> None:
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="result",
            parts=[Part(root=TextPart(text="Test"))],
            metadata={CONTEXT_KEY: {"session": "123"}},
        )
        result = response_message_from_a2a_artifacts([artifact])

        assert result == ResponseMessage(
            messages=[{"content": "Test"}],
            context={"session": "123"},
        )

    def test_multiple_artifacts_raises_error(self) -> None:
        artifacts = [
            Artifact(
                artifact_id=str(uuid4()),
                name="art1",
                parts=[Part(root=TextPart(text="Text 1"))],
            ),
            Artifact(
                artifact_id=str(uuid4()),
                name="art2",
                parts=[Part(root=TextPart(text="Text 2"))],
            ),
        ]

        with pytest.raises(NotImplementedError, match="Multiple artifacts are not supported"):
            response_message_from_a2a_artifacts(artifacts)

    def test_multiple_parts_raises_error(self) -> None:
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="result",
            parts=[
                Part(root=TextPart(text="Part 1")),
                Part(root=TextPart(text="Part 2")),
            ],
        )

        with pytest.raises(NotImplementedError, match="Multiple parts are not supported"):
            response_message_from_a2a_artifacts([artifact])


class TestResponseMessageFromA2ATask:
    def test_task_input_required_with_history(self) -> None:
        task = Task(
            id=str(uuid4()),
            context_id=str(uuid4()),
            status=TaskStatus(state=TaskState.input_required),
            history=[
                Message(
                    role=Role.agent,
                    parts=[Part(root=TextPart(text="Hi, user! Please provide input"))],
                    message_id=str(uuid4()),
                )
            ],
            artifacts=[],
        )

        result = response_message_from_a2a_task(task)

        assert result == ResponseMessage(
            messages=[{"content": "Hi, user! Please provide input", "role": "assistant"}],
            input_required="Hi, user! Please provide input",
        )

    def test_task_input_required_with_empty_history(self) -> None:
        task = Task(
            id=str(uuid4()),
            context_id=str(uuid4()),
            status=TaskStatus(state=TaskState.input_required),
            history=[],
            artifacts=[],
        )

        result = response_message_from_a2a_task(task)

        assert result == ResponseMessage(input_required="Please provide input:")

    def test_task_completed_with_artifacts(self) -> None:
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="result",
            parts=[Part(root=TextPart(text="Task completed"))],
        )
        task = Task(
            id=str(uuid4()),
            context_id=str(uuid4()),
            status=TaskStatus(state=TaskState.completed),
            history=[],
            artifacts=[artifact],
        )

        result = response_message_from_a2a_task(task)

        assert result == ResponseMessage(messages=[{"content": "Task completed"}])

    def test_task_completed_with_no_artifacts(self) -> None:
        task = Task(
            id=str(uuid4()),
            context_id=str(uuid4()),
            status=TaskStatus(state=TaskState.completed),
            history=[],
            artifacts=[],
        )

        result = response_message_from_a2a_task(task)

        assert result is None

    def test_task_completed_with_artifact_context(self) -> None:
        artifact = Artifact(
            artifact_id=str(uuid4()),
            name="result",
            parts=[Part(root=TextPart(text="Result with context"))],
            metadata={CONTEXT_KEY: {"session": "xyz"}},
        )
        task = Task(
            id=str(uuid4()),
            context_id=str(uuid4()),
            status=TaskStatus(state=TaskState.completed),
            history=[],
            artifacts=[artifact],
        )

        result = response_message_from_a2a_task(task)

        assert result == ResponseMessage(
            messages=[{"content": "Result with context"}],
            context={"session": "xyz"},
        )


class TestResponseMessageFromA2AMessage:
    def test_empty_message(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[],
            message_id=str(uuid4()),
        )
        result = response_message_from_a2a_message(message)

        assert result == ResponseMessage(messages=[{"content": ""}])

    def test_single_text_part(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[Part(root=TextPart(text="Hello"))],
            message_id=str(uuid4()),
        )
        result = response_message_from_a2a_message(message)

        assert result == ResponseMessage(messages=[{"content": "Hello"}])

    def test_multiple_text_parts(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[
                Part(root=TextPart(text="Hello")),
                Part(root=TextPart(text="World")),
                Part(root=TextPart(text="!")),
            ],
            message_id=str(uuid4()),
        )
        result = response_message_from_a2a_message(message)

        assert result == ResponseMessage(messages=[{"content": "Hello\nWorld\n!"}])

    def test_single_data_part(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[Part(root=DataPart(data={"key": "value"}))],
            message_id=str(uuid4()),
        )
        result = response_message_from_a2a_message(message)

        assert result == ResponseMessage(
            messages=[{"key": "value"}],
        )

    def test_mixed_text_and_data_parts(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[
                Part(root=TextPart(text="Text 1")),
                Part(root=DataPart(data={"result": "42"})),
            ],
            message_id=str(uuid4()),
        )

        with pytest.raises(NotImplementedError, match="Data parts and text parts are not supported together"):
            response_message_from_a2a_message(message)

    def test_multiple_data_parts_raises_error(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[
                Part(root=DataPart(data={"key1": "value1"})),
                Part(root=DataPart(data={"key2": "value2"})),
            ],
            message_id=str(uuid4()),
        )

        with pytest.raises(NotImplementedError, match="Multiple data parts are not supported"):
            response_message_from_a2a_message(message)

    def test_message_with_context(self) -> None:
        message = Message(
            role=Role.agent,
            parts=[],
            message_id=str(uuid4()),
            metadata={CONTEXT_KEY: {"session_id": "abc123"}},
        )
        response = response_message_from_a2a_message(message)

        assert response == ResponseMessage(
            messages=[{"content": ""}],
            context={"session_id": "abc123"},
        )


class TestResponseMessageToA2A:
    def test_none_response(self) -> None:
        artifact, messages, input_required = response_message_to_a2a(None, "ctx-123", "task-456")

        assert not input_required
        assert artifact == Artifact(
            name="result",
            parts=[],
            # randomly generated artifact_id
            artifact_id=artifact.artifact_id,
        )
        assert messages == []

    def test_response_with_context(self) -> None:
        response = ResponseMessage(
            messages=[{"content": "Hello"}],
            context={"key": "value"},
        )
        artifact, messages, input_required = response_message_to_a2a(response, "ctx-123", "task-456")

        assert not input_required

        assert artifact == Artifact(
            name="result",
            parts=[Part(root=TextPart(text="Hello"))],
            metadata={CONTEXT_KEY: {"key": "value"}},
            # randomly generated artifact_id
            artifact_id=artifact.artifact_id,
        )

        assert messages == [
            Message(
                context_id="ctx-123",
                task_id="task-456",
                parts=[
                    Part(root=TextPart(text="Hello")),
                ],
                role=Role.agent,
                # randomly generated message_id
                message_id=messages[0].message_id,
            )
        ]

    def test_response_with_multiple_messages(self) -> None:
        response = ResponseMessage(
            messages=[
                {"content": "Message 1"},
                {"content": "Message 2"},
                {"content": "Message 3"},
            ]
        )
        artifact, messages, input_required = response_message_to_a2a(response, "ctx-123", "task-456")

        assert not input_required

        # Artifact should contain only the last message
        assert artifact == Artifact(
            name="result",
            parts=[Part(root=TextPart(text="Message 3"))],
            # randomly generated artifact_id
            artifact_id=artifact.artifact_id,
        )

        # History messages should contain all message with all history parts
        assert len(messages) == 1
        assert len(messages[0].parts) == 3


class TestRoundTripConversions:
    def test_request_round_trip(self) -> None:
        original_request = RequestMessage(
            messages=[{"content": "Hello", "role": "user"}],
            context={"key": "value"},
            client_tools=[{"function": {"name": "tool1"}}],
        )

        a2a_message = request_message_to_a2a(original_request, str(uuid4()))
        converted_request = request_message_from_a2a(a2a_message)

        assert converted_request == original_request

    def test_response_round_trip_with_message(self) -> None:
        original_response = ResponseMessage(
            messages=[{"content": "Response text"}],
            context={"session": "123"},
        )

        # Convert to A2A and back
        artifact, _, _ = response_message_to_a2a(original_response, "ctx-123", "task-456")
        converted_response = response_message_from_a2a_artifacts([artifact])

        assert converted_response == original_response
