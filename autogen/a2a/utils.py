# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast
from uuid import uuid4

from a2a.types import Artifact, DataPart, Message, Part, Role, TextPart
from a2a.utils import new_agent_parts_message, new_artifact

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
) -> tuple[Artifact, list[Message]]:
    # mypy ignores could be removed after
    # https://github.com/a2aproject/a2a-python/pull/503

    if not result:
        return new_artifact(
            name="result",
            parts=[],
            description=None,  # type: ignore[arg-type]
        ), []

    artifact = new_artifact(
        name="result",
        parts=[message_to_part(result.messages[-1])],
        description=None,  # type: ignore[arg-type]
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
