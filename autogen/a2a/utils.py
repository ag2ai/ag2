# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast
from uuid import uuid4

from a2a.types import Artifact, DataPart, Message, Part, Role, Task, TaskState, TextPart
from a2a.utils import get_message_text, new_agent_parts_message, new_artifact
from a2a.utils.message import new_agent_text_message

from autogen.remote.protocol import RequestMessage, ResponseMessage

AG2_METADATA_KEY_PREFIX = "ag2_"
CLIENT_TOOLS_KEY = f"{AG2_METADATA_KEY_PREFIX}client_tools"
CONTEXT_KEY = f"{AG2_METADATA_KEY_PREFIX}context_update"


def request_message_to_a2a(
    request_message: RequestMessage,
    context_id: str,
) -> Message:
    metadata: dict[str, Any] = {}
    if request_message.client_tools:
        metadata[CLIENT_TOOLS_KEY] = request_message.client_tools
    if request_message.context:
        metadata[CONTEXT_KEY] = request_message.context

    return Message(
        role=Role.user,
        parts=[message_to_part(message) for message in request_message.messages],
        message_id=uuid4().hex,
        context_id=context_id,
        metadata=metadata,
    )


def request_message_from_a2a(message: Message) -> RequestMessage:
    metadata = message.metadata or {}
    return RequestMessage(
        messages=[message_from_part(part) for part in message.parts],
        context=metadata.get(CONTEXT_KEY),
        client_tools=metadata.get(CLIENT_TOOLS_KEY, []),
    )


def response_message_from_a2a_task(task: Task) -> ResponseMessage | None:
    history = [message_from_part(p) for m in (task.history or []) for p in m.parts]

    if task.status.state is TaskState.input_required:
        message: str | None = None
        context: dict[str, Any] | None = None

        if task.history:
            input_message = task.history[-1]
            message = get_message_text(input_message)

            if input_message.metadata:
                context = input_message.metadata.get(CONTEXT_KEY)

            if "role" not in history[-1]:
                history[-1]["role"] = "assistant"

        return ResponseMessage(
            messages=history,
            input_required=message or "Please provide input:",
            context=context,
        )

    response = response_message_from_a2a_artifacts(task.artifacts)
    if response:
        response.messages = history + response.messages
    return response


def response_message_from_a2a_artifacts(artifacts: list[Artifact] | None) -> ResponseMessage | None:
    if not artifacts:
        return None

    if len(artifacts) > 1:
        raise NotImplementedError("Multiple artifacts are not supported")

    artifact = artifacts[-1]

    if not artifact.parts:
        return None

    if len(artifact.parts) > 1:
        raise NotImplementedError("Multiple parts are not supported")

    return ResponseMessage(
        messages=[message_from_part(artifact.parts[-1])],
        context=(artifact.metadata or {}).get(CONTEXT_KEY),
    )


def response_message_from_a2a_message(message: Message) -> ResponseMessage | None:
    text_parts: list[Part] = []
    data_parts: list[Part] = []
    for part in message.parts:
        if isinstance(part.root, TextPart):
            text_parts.append(part)
        elif isinstance(part.root, DataPart):
            data_parts.append(part)
        else:
            raise NotImplementedError(f"Unsupported part type: {type(part.root)}")

    tpn = len(text_parts)
    if dpn := len(data_parts):
        if dpn > 1:
            raise NotImplementedError("Multiple data parts are not supported")

        if tpn:
            raise NotImplementedError("Data parts and text parts are not supported together")

        messages = [message_from_part(data_parts[0])]
    elif tpn == 1:
        messages = [message_from_part(text_parts[0])]
    else:
        messages = [{"content": "\n".join(cast(TextPart, t.root).text for t in text_parts)}]

    return ResponseMessage(
        messages=messages,
        context=(message.metadata or {}).get(CONTEXT_KEY),
    )


def response_message_to_a2a(
    result: ResponseMessage | None,
    context_id: str | None,
    task_id: str | None,
) -> tuple[Artifact | None, list[Message], Message | None]:
    if not result:
        return (
            new_artifact(
                name="result",
                parts=[],
                description=None,
            ),
            [],
            None,
        )

    message_history = [
        new_agent_parts_message(
            parts=[message_to_part(m) for m in result.messages],
            context_id=context_id,
            task_id=task_id,
        ),
    ]

    if result.input_required is not None:
        input_message = new_agent_text_message(
            text=result.input_required,
            context_id=context_id,
            task_id=task_id,
        )
        if result.context:
            input_message.metadata = {CONTEXT_KEY: result.context}

        return None, message_history, input_message

    artifact = new_artifact(
        name="result",
        parts=[message_to_part(result.messages[-1])],
        description=None,
    )

    if result.context:
        artifact.metadata = {CONTEXT_KEY: result.context}

    return artifact, message_history, None


def message_to_part(message: dict[str, Any]) -> Part:
    message = message.copy()
    text = message.pop("content", "") or ""
    return Part(
        root=TextPart(
            text=text,
            metadata=message or None,
        )
    )


def message_from_part(part: Part) -> dict[str, Any]:
    root = part.root

    if isinstance(root, TextPart):
        return {
            **(root.metadata or {}),
            "content": root.text,
        }

    elif isinstance(root, DataPart):
        if (  # pydantic-ai specific
            set(root.data.keys()) == {"result"}
            and root.metadata
            and "json_schema" in root.metadata
            and isinstance(data := root.data["result"], dict)
        ):
            return data

        return root.data

    else:
        raise NotImplementedError(f"Unsupported part type: {type(part.root)}")
