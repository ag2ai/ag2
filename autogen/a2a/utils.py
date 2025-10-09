# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import uuid4

from a2a.types import Artifact, DataPart, Message, Part, Role, TextPart
from a2a.utils import get_message_text, new_agent_parts_message, new_artifact

from autogen.remote.protocol import RequestMessage, ResponseMessage

CLIENT_TOOLS_KEY = "ag2_client_tools"
CONTEXT_KEY = "ag2_context_update"


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


def response_message_from_a2a(artifacts: list[Artifact] | None) -> ResponseMessage | None:
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
    return ResponseMessage(
        messages=[{"content": get_message_text(message)}],
        context=(message.metadata or {}).get(CONTEXT_KEY),
    )


def response_message_to_a2a(
    result: ResponseMessage | None,
    context_id: str | None,
    task_id: str | None,
) -> tuple[Artifact, list[Message]]:
    if not result:
        return new_artifact(name="result", parts=[]), []

    artifact = new_artifact(
        name="result",
        parts=[message_to_part(result.messages[-1])],
    )

    if result.context:
        artifact.metadata = {CONTEXT_KEY: result.context}

    return (
        artifact,
        [
            new_agent_parts_message(
                parts=[message_to_part(m) for m in result.messages],
                context_id=context_id,
                task_id=task_id,
            ),
        ],
    )


def message_to_part(message: dict[str, Any]) -> Part:
    text = message.pop("content", "")
    return Part(root=TextPart(text=text, metadata=message))


def message_from_part(part: Part) -> dict[str, Any]:
    if isinstance(part.root, TextPart):
        return {
            **(part.root.metadata or {}),
            "content": part.root.text,
        }

    elif isinstance(part.root, DataPart):
        if set(part.root.data.keys()) == {"result"} and "":  # pydantic-ai specific
            return part.root.data["result"]

        return part.root.data

    else:
        raise NotImplementedError(f"Unsupported part type: {type(part.root)}")
