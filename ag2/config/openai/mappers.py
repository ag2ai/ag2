# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, TypeVar, cast

from fast_depends.library.serializer import SerializerProto
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_param import File
from openai.types.responses import (
    EasyInputMessageParam,
    FileSearchToolParam,
    FunctionShellToolParam,
    ImageDetail,
    ResponseFunctionCallOutputItemParam,
    ResponseFunctionShellToolCall,
    ResponseFunctionToolCallParam,
    ResponseInputContentParam,
    ResponseInputFileContentParam,
    ResponseInputFileParam,
    ResponseInputImageContentParam,
    ResponseInputImageParam,
    ResponseInputItemParam,
    ResponseInputTextContentParam,
    ResponseInputTextParam,
    ResponseUsage,
    SkillReferenceParam,
    ToolSearchToolParam,
    WebSearchToolParam,
)
from openai.types.responses.container_auto_param import ContainerAutoParam
from openai.types.responses.container_reference_param import ContainerReferenceParam
from openai.types.responses.file_search_tool_param import Filters as FileSearchFilters
from openai.types.responses.function_shell_tool_param import Environment as ShellEnvironmentParam
from openai.types.responses.local_environment_param import LocalEnvironmentParam
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.tool_param import (
    CodeInterpreter,
    CodeInterpreterContainerCodeInterpreterToolAuto,
    ImageGeneration,
    Mcp,
)
from openai.types.responses.web_search_tool_param import UserLocation as WebSearchUserLocation

from ag2.compact import CompactionSummary
from ag2.config.openai.events import (
    OpenAIReasoningEvent,
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
)
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolResultsEvent,
    UrlInput,
    Usage,
)
from ag2.exceptions import (
    BlockedToolsUnsupportedError,
    ClientExecutedShellUnsupportedError,
    UnsupportedInputError,
    UnsupportedToolError,
)
from ag2.files.types import FileProvider
from ag2.response import ResponseProto
from ag2.tools.builtin.code_execution import CodeExecutionToolSchema
from ag2.tools.builtin.file_search import FileSearchToolSchema
from ag2.tools.builtin.image_generation import ImageGenerationToolSchema
from ag2.tools.builtin.mcp_server import MCPServerToolSchema
from ag2.tools.builtin.shell import (
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    ShellEnvironment,
    ShellToolSchema,
)
from ag2.tools.builtin.skills import SkillsToolSchema
from ag2.tools.builtin.tool_search import ToolSearchToolSchema
from ag2.tools.builtin.web_search import UserLocation, WebSearchToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

_Detail = TypeVar("_Detail", bound=str)

_OUTPUT_ONLY_FIELDS = {"created_by"}
"""Fields the API puts on a hosted output item but rejects on the way back in.

``created_by`` is answered with ``Unknown parameter: input[N].created_by``. The API leaves
it unset today, so ``exclude_none`` already drops it; this is the guard for when it does not.
"""


def _kind_label(kind: BinaryType | str) -> str:
    """Kind for error messages; may be a raw string in logs persisted before the ``__enum__`` marker."""
    return kind.value if isinstance(kind, BinaryType) else str(kind)


def response_proto_to_schema(response: ResponseProto | None) -> dict[str, Any] | None:
    """Convert a ResponseProto to Chat Completions response_format."""
    if not response or not response.json_schema:
        return None

    strict_schema = _strictify_schema(response.json_schema)
    schema: dict[str, Any] = {
        "schema": strict_schema,
        "name": response.name,
        "strict": True,
    }
    if response.description:
        schema["description"] = response.description

    return {"type": "json_schema", "json_schema": schema}


def _strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively coerce a JSON Schema into OpenAI's strict subset."""
    schema = dict(schema)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)

    if "properties" in schema:
        schema["properties"] = {
            k: _strictify_schema(v) if isinstance(v, dict) else v for k, v in schema["properties"].items()
        }
        schema["required"] = list(schema["properties"])

    if "$defs" in schema:
        schema["$defs"] = {k: _strictify_schema(v) if isinstance(v, dict) else v for k, v in schema["$defs"].items()}

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [_strictify_schema(item) if isinstance(item, dict) else item for item in schema[key]]

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _strictify_schema(schema["items"])

    return schema


def response_proto_to_text_config(
    response: ResponseProto | None,
) -> dict[str, Any] | None:
    """Convert a ResponseProto to Responses API text config."""
    if not response or not response.json_schema:
        return None

    strict_schema = _strictify_schema(response.json_schema)

    fmt: dict[str, Any] = {
        "type": "json_schema",
        "name": response.name,
        "schema": strict_schema,
        "strict": True,
    }
    if response.description:
        fmt["description"] = response.description

    return {"format": fmt}


def events_to_responses_input(
    messages: Sequence[BaseEvent],
    serializer: SerializerProto,
) -> list[ResponseInputItemParam]:
    """Convert a sequence of events to Responses API input items."""
    result: list[ResponseInputItemParam] = []
    seen_reasoning_ids: set[str] = set()
    answered_shell_calls = _answered_shell_calls(messages)

    for message in messages:
        if isinstance(message, ModelResponse):
            if message.message:
                result.append(_easy_message("assistant", message.message.content))
            # Add function call items from the response
            for call in message.tool_calls.calls:
                function_call: ResponseFunctionToolCallParam = {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                result.append(function_call)

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                blocks: list[ResponseFunctionCallOutputItemParam] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        blocks.append(_tool_output_text(part.content))
                    elif isinstance(part, DataInput):
                        blocks.append(_tool_output_text(serializer.encode(part.data).decode()))
                    elif isinstance(part, BinaryInput):
                        b64 = base64.b64encode(part.data).decode()
                        if part.kind == BinaryType.IMAGE:
                            # Images in output must use input_image (input_file rejects image/* MIME).
                            output_image: ResponseInputImageContentParam = {
                                "type": "input_image",
                                "image_url": f"data:{part.media_type};base64,{b64}",
                            }
                            blocks.append(output_image)
                        elif part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                            # input_file with file_data *requires* filename.
                            output_file: ResponseInputFileContentParam = {
                                "type": "input_file",
                                "file_data": f"data:{part.media_type};base64,{b64}",
                                "filename": _filename(part),
                            }
                            blocks.append(output_file)
                        else:
                            raise UnsupportedInputError(f"BinaryInput({_kind_label(part.kind)})", "openai-responses")
                    elif isinstance(part, UrlInput):
                        if part.kind == BinaryType.IMAGE:
                            url_image: ResponseInputImageContentParam = {"type": "input_image", "image_url": part.url}
                            blocks.append(url_image)
                        elif part.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                            # file_url forbids filename (API mutual-exclusion).
                            url_file: ResponseInputFileContentParam = {"type": "input_file", "file_url": part.url}
                            blocks.append(url_file)
                        else:
                            raise UnsupportedInputError(f"UrlInput({_kind_label(part.kind)})", "openai-responses")
                    elif isinstance(part, FileIdInput):
                        # file_id forbids filename in output (user-message allows both).
                        file_ref: ResponseInputFileContentParam = {"type": "input_file", "file_id": part.file_id}
                        blocks.append(file_ref)
                    else:
                        raise UnsupportedInputError(type(part).__name__, "openai-responses")

                output: FunctionCallOutput = {
                    "type": "function_call_output",
                    "call_id": r.parent_id,
                    "output": block["text"]
                    if len(blocks) == 1 and (block := blocks[0])["type"] == "input_text"
                    else blocks,
                }
                result.append(output)

        elif isinstance(message, OpenAIReasoningEvent):
            if message.item.id not in seen_reasoning_ids:
                seen_reasoning_ids.add(message.item.id)
                result.append(_replayed_item(message.item.model_dump(exclude_none=True, mode="json")))

        elif isinstance(message, OpenAIServerToolCallEvent):
            # A `shell_call` is the one hosted item the API will not accept
            # alone: its outcome lives in a separate `shell_call_output`. A turn
            # that ended between the two — an `incomplete` response, say — leaves
            # the call unanswered, and replaying it would 400 the next request.
            if (
                isinstance(message.item, ResponseFunctionShellToolCall)
                and message.item.call_id not in answered_shell_calls
            ):
                continue

            # warnings=False: openai SDK pins ActionSearchSource.type to
            # Literal["url"] but the API returns other values (e.g. "api"),
            # which makes pydantic warn on every round-trip serialization.
            result.append(
                _replayed_item(
                    message.item.model_dump(exclude_none=True, mode="json", warnings=False, exclude=_OUTPUT_ONLY_FIELDS)
                )
            )

        elif isinstance(message, OpenAIServerToolResultEvent) and message.item is not None:
            # Only a hosted shell call carries one: its output is a separate
            # item, so the call replays incomplete without it.
            result.append(
                _replayed_item(message.item.model_dump(exclude_none=True, mode="json", exclude=_OUTPUT_ONLY_FIELDS))
            )

        elif isinstance(message, ModelRequest):
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    result.append(_user_message(_input_text(inp.content)))

                elif isinstance(inp, DataInput):
                    result.append(_user_message(_input_text(serializer.encode(inp.data).decode())))

                elif isinstance(inp, FileIdInput):
                    if (provider := getattr(inp, "provider", None)) and provider is not FileProvider.OPENAI:
                        raise UnsupportedInputError(
                            f"file uploaded via '{provider.value}' cannot be used with '{FileProvider.OPENAI.value}'",
                            "openai-responses",
                        )
                    # OpenAI Responses API: file_id and filename are mutually exclusive.
                    # filename applies to inline file_data, not to file_id references.
                    file_id_input: ResponseInputFileParam = {"type": "input_file", "file_id": inp.file_id}
                    result.append(_user_message(file_id_input))

                elif isinstance(inp, BinaryInput):
                    b64 = base64.b64encode(inp.data).decode()
                    if inp.kind == BinaryType.IMAGE:
                        detail = _image_detail(inp, _RESPONSES_IMAGE_DETAILS, "openai-responses")
                        result.append(_user_message(_input_image(f"data:{inp.media_type};base64,{b64}", detail)))

                    elif inp.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                        # input_file with file_data *requires* filename.
                        file_data_input: ResponseInputFileParam = {
                            "type": "input_file",
                            "file_data": f"data:{inp.media_type};base64,{b64}",
                            "filename": _filename(inp),
                        }
                        result.append(_user_message(file_data_input))

                    else:
                        raise UnsupportedInputError(f"BinaryInput({_kind_label(inp.kind)})", "openai-responses")

                elif isinstance(inp, UrlInput):
                    if inp.kind == BinaryType.IMAGE:
                        result.append(_user_message(_input_image(inp.url, None)))

                    elif inp.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
                        file_url_input: ResponseInputFileParam = {"type": "input_file", "file_url": inp.url}
                        result.append(_user_message(file_url_input))

                    else:
                        raise UnsupportedInputError(f"UrlInput({_kind_label(inp.kind)})", "openai-responses")

                else:
                    raise UnsupportedInputError(type(inp).__name__, "openai-responses")

        elif isinstance(message, CompactionSummary):
            # Surface the summary as a user turn so it stays visible and gives a valid opening turn
            text = f"[Summary of earlier conversation]\n{message.summary}"
            result.append(_user_message(_input_text(text)))

    return result


def _easy_message(
    role: Literal["user", "assistant"], content: str | list[ResponseInputContentParam]
) -> EasyInputMessageParam:
    return {"role": role, "content": content}


def _user_message(part: ResponseInputContentParam) -> EasyInputMessageParam:
    return _easy_message("user", [part])


def _input_text(text: str) -> ResponseInputTextParam:
    return {"type": "input_text", "text": text}


def _tool_output_text(text: str) -> ResponseInputTextContentParam:
    return {"type": "input_text", "text": text}


def _input_image(image_url: str, detail: ImageDetail | None) -> ResponseInputImageParam:
    if detail is not None:
        return {"type": "input_image", "image_url": image_url, "detail": detail}
    # The SDK marks `detail` Required, but the API accepts its omission (verified live) and
    # applies its own default; ag2 does not send a detail the caller did not ask for.
    return cast(ResponseInputImageParam, {"type": "input_image", "image_url": image_url})


def _replayed_item(item: dict[str, Any]) -> ResponseInputItemParam:
    # The one untyped boundary: the dump of the SDK's own output item, which pydantic
    # validated on the way in, is the input item of the same kind.
    return cast(ResponseInputItemParam, item)


def convert_messages(
    system_prompt: Iterable[str],
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[ChatCompletionMessageParam]:
    # legacy prompt message format
    system: ChatCompletionSystemMessageParam = {"content": "\n".join(system_prompt), "role": "system"}
    result: list[ChatCompletionMessageParam] = [system]

    for message in messages:
        if isinstance(message, ModelResponse):
            result.append(_assistant_message(message))

        elif isinstance(message, ToolResultsEvent):
            for r in message.results:
                result_parts: list[ChatCompletionContentPartTextParam] = []
                for part in r.result.parts:
                    if isinstance(part, TextInput):
                        result_parts.append({"type": "text", "text": part.content})
                    elif isinstance(part, DataInput):
                        result_parts.append({"type": "text", "text": serializer.encode(part.data).decode()})
                    else:
                        raise UnsupportedInputError(type(part).__name__, "openai-completions")

                tool_message: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "tool_call_id": r.parent_id,
                    # Simple string content for a single plain-text turn (most common case)
                    "content": result_parts[0]["text"] if len(result_parts) == 1 else result_parts,
                }
                result.append(tool_message)

        elif isinstance(message, ModelRequest):
            parts: list[ChatCompletionContentPartParam] = []
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    parts.append(_chat_text_part(inp.content))

                elif isinstance(inp, DataInput):
                    parts.append(_chat_text_part(serializer.encode(inp.data).decode()))

                elif isinstance(inp, UrlInput):
                    if inp.kind == BinaryType.IMAGE:
                        url_image: ChatCompletionContentPartImageParam = {
                            "type": "image_url",
                            "image_url": {"url": inp.url},
                        }
                        parts.append(url_image)

                    else:
                        raise UnsupportedInputError(f"UrlInput({_kind_label(inp.kind)})", "openai-completions")

                elif isinstance(inp, FileIdInput):
                    file_ref: File = {"type": "file", "file": {"file_id": inp.file_id}}
                    parts.append(file_ref)

                elif isinstance(inp, BinaryInput):
                    if inp.kind == BinaryType.AUDIO:
                        audio_format = _MIME_TO_AUDIO_FORMAT.get(inp.media_type)
                        if audio_format is None:
                            # The API answers anything else with `Supported values are: 'wav' and 'mp3'`.
                            raise UnsupportedInputError(
                                f"BinaryInput(audio, media_type={inp.media_type})", "openai-completions"
                            )
                        audio: ChatCompletionContentPartInputAudioParam = {
                            "type": "input_audio",
                            "input_audio": {"data": base64.b64encode(inp.data).decode(), "format": audio_format},
                        }
                        parts.append(audio)

                    elif inp.kind == BinaryType.IMAGE:
                        b64 = base64.b64encode(inp.data).decode()
                        image_url: ImageURL = {"url": f"data:{inp.media_type};base64,{b64}"}
                        if (detail := _image_detail(inp, _CHAT_IMAGE_DETAILS, "openai-completions")) is not None:
                            image_url["detail"] = detail
                        image: ChatCompletionContentPartImageParam = {"type": "image_url", "image_url": image_url}
                        parts.append(image)

                    elif inp.kind == BinaryType.DOCUMENT:
                        b64 = base64.b64encode(inp.data).decode()
                        document: File = {
                            "type": "file",
                            "file": {"file_data": f"data:{inp.media_type};base64,{b64}", "filename": _filename(inp)},
                        }
                        parts.append(document)

                    else:
                        raise UnsupportedInputError(f"BinaryInput({_kind_label(inp.kind)})", "openai-completions")

                else:
                    raise UnsupportedInputError(type(inp).__name__, "openai-completions")

            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                # Simple string content for a single plain-text turn (most common case)
                "content": parts[0]["text"] if len(parts) == 1 and parts[0]["type"] == "text" else parts,
            }
            result.append(user_message)

        elif isinstance(message, CompactionSummary):
            # Surface the summary as a user turn so it stays visible and gives a valid opening turn
            summary: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": f"[Summary of earlier conversation]\n{message.summary}",
            }
            result.append(summary)

    return result


def _assistant_message(message: ModelResponse) -> ChatCompletionAssistantMessageParam:
    """Replay a model turn: its text, then the function calls it made."""
    assistant: ChatCompletionAssistantMessageParam = {
        "content": message.message.content if message.message else None,
        "role": "assistant",
    }
    if message.tool_calls:
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"arguments": json.dumps(call.serialized_arguments), "name": call.name},
            }
            for call in message.tool_calls.calls
        ]
        assistant["tool_calls"] = tool_calls
    return assistant


def _chat_text_part(text: str) -> ChatCompletionContentPartTextParam:
    return {"type": "text", "text": text}


def _image_detail(inp: BinaryInput, details: Mapping[str, _Detail], provider: str) -> _Detail | None:
    """The ``detail`` a caller set in ``vendor_metadata``, refused unless the API knows it."""
    if "detail" not in inp.vendor_metadata:
        return None
    detail = inp.vendor_metadata["detail"]
    if not isinstance(detail, str) or detail not in details:
        raise UnsupportedInputError(f"BinaryInput(image, detail={detail!r})", provider)
    return details[detail]


def _filename(inp: BinaryInput) -> str:
    """The name inline file data travels under; the API requires one."""
    filename = inp.vendor_metadata.get("filename")
    if filename:
        return str(filename)
    suffix = inp.media_type.rsplit("/", 1)[-1].split("+", 1)[0]
    return f"file.{suffix}"


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """OpenAI requires tool parameters to be type: object with properties."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    schema.setdefault("additionalProperties", False)
    return schema


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    """Chat Completions API tool format."""
    if isinstance(t, FunctionToolSchema):
        if t.defer_loading:
            # Tool search / deferred loading is a Responses-API feature; the
            # Chat Completions API has no way to load deferred tools. Fail fast
            # instead of silently sending the tool eagerly (which would defeat
            # defer_loading and give no error). Use the Responses API instead.
            raise UnsupportedToolError("function with defer_loading (use the Responses API)", "openai-completions")
        fn_tool: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": t.function.name,
                "description": t.function.description,
                "parameters": _ensure_object_schema(t.function.parameters),
            },
        }
        return dict(fn_tool)

    raise UnsupportedToolError(t.type, "openai-completions")


def _user_location_to_api(location: UserLocation) -> WebSearchUserLocation:
    """Tag the location the way the API discriminates it; only the fields that were set travel."""
    result: WebSearchUserLocation = {"type": "approximate"}
    if location.city is not None:
        result["city"] = location.city
    if location.region is not None:
        result["region"] = location.region
    if location.country is not None:
        result["country"] = location.country
    if location.timezone is not None:
        result["timezone"] = location.timezone
    return result


def _shell_environment_to_api(environment: ShellEnvironment) -> ShellEnvironmentParam:
    """Map the container the hosted shell runs in. An unrecognised environment runs locally."""
    if isinstance(environment, ContainerAutoEnvironment):
        container_auto: ContainerAutoParam = {"type": "container_auto"}
        if environment.network_policy is not None:
            container_auto["network_policy"] = {
                "type": "allowlist",
                "allowed_domains": environment.network_policy.allowed_domains,
            }
        return container_auto

    if isinstance(environment, ContainerReferenceEnvironment):
        container_reference: ContainerReferenceParam = {
            "type": "container_reference",
            "container_id": environment.container_id,
        }
        return container_reference

    local: LocalEnvironmentParam = {"type": "local"}
    return local


def tool_to_responses_api(t: ToolSchema) -> dict[str, Any]:
    """Responses API tool format — name/description at top level."""
    if isinstance(t, FunctionToolSchema):
        # Built by hand, unlike its neighbours: the SDK's `FunctionToolParam` makes `strict`
        # required-but-nullable, and ag2 omits the key so the API applies its own default.
        fn_tool: dict[str, Any] = {
            "type": "function",
            "name": t.function.name,
            "description": t.function.description,
            "parameters": _ensure_object_schema(t.function.parameters),
        }
        if t.defer_loading:
            fn_tool["defer_loading"] = True
        return fn_tool

    elif isinstance(t, WebSearchToolSchema):
        web_search: WebSearchToolParam = {"type": "web_search"}
        if t.search_context_size is not None:
            web_search["search_context_size"] = t.search_context_size
        if t.user_location is not None:
            web_search["user_location"] = _user_location_to_api(t.user_location)
        if t.allowed_domains is not None:
            web_search["filters"] = {"allowed_domains": t.allowed_domains}

        web_search_entry = dict(web_search)
        if t.max_uses is not None:
            # `max_uses` is Anthropic's cap. The SDK's WebSearchToolParam has no such field and
            # OpenAI documents none, but ag2 has always sent it, so this refactor keeps sending it.
            web_search_entry["max_uses"] = t.max_uses
        return web_search_entry

    elif isinstance(t, FileSearchToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-file-search
        file_search: FileSearchToolParam = {"type": "file_search", "vector_store_ids": t.vector_store_ids}
        if t.max_num_results is not None:
            file_search["max_num_results"] = t.max_num_results
        if t.filters is not None:
            # A comparison or compound filter, authored as JSON by the caller.
            file_search["filters"] = cast(FileSearchFilters, t.filters)
        return dict(file_search)

    elif isinstance(t, CodeExecutionToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-code-interpreter
        container: CodeInterpreterContainerCodeInterpreterToolAuto = {"type": "auto"}
        code_interpreter: CodeInterpreter = {"type": "code_interpreter", "container": container}
        return dict(code_interpreter)

    elif isinstance(t, ShellToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-shell
        shell: FunctionShellToolParam = {"type": "shell"}
        if t.environment is not None:
            shell["environment"] = _shell_environment_to_api(t.environment)
        return dict(shell)

    elif isinstance(t, ImageGenerationToolSchema):
        image_generation: ImageGeneration = {"type": "image_generation"}
        if t.quality is not None:
            image_generation["quality"] = t.quality
        if t.size is not None:
            image_generation["size"] = t.size
        if t.background is not None:
            image_generation["background"] = t.background
        if t.output_format is not None:
            image_generation["output_format"] = t.output_format
        if t.output_compression is not None:
            image_generation["output_compression"] = t.output_compression
        if t.partial_images is not None:
            image_generation["partial_images"] = t.partial_images
        return dict(image_generation)

    elif isinstance(t, MCPServerToolSchema):
        if t.blocked_tools is not None:
            raise BlockedToolsUnsupportedError("the OpenAI Responses API", t.server_label)

        # https://platform.openai.com/docs/guides/tools-remote-mcp
        mcp: Mcp = {
            "type": "mcp",
            "server_label": t.server_label,
            "server_url": t.server_url,
            "require_approval": "never",
        }
        if t.description is not None:
            mcp["server_description"] = t.description

        if t.allowed_tools is not None:
            mcp["allowed_tools"] = t.allowed_tools
        if t.headers is not None:
            mcp["headers"] = t.headers
        elif t.authorization_token is not None:
            mcp["headers"] = {"Authorization": f"Bearer {t.authorization_token}"}
        return dict(mcp)

    elif isinstance(t, SkillsToolSchema):
        # Skills never appear directly in tools[] — the Responses client extracts
        # them via extract_skills_for_shell() and attaches them to the hosted
        # shell tool's container environment (merge_skills_into_shell_tools).
        raise UnsupportedToolError(t.type, "openai-responses")

    elif isinstance(t, ToolSearchToolSchema):
        # https://developers.openai.com/api/docs/guides/tools-tool-search
        # OpenAI exposes a single server-side tool-search tool; mode is Anthropic-only.
        tool_search: ToolSearchToolParam = {"type": "tool_search"}
        return dict(tool_search)

    raise UnsupportedToolError(t.type, "openai-responses")


def extract_skills_for_shell(tools: Iterable[ToolSchema]) -> list[SkillReferenceParam]:
    """Extract OpenAI ``skill_reference`` entries from SkillsToolSchema instances.

    OpenAI Responses attaches skills to the hosted shell tool's container
    environment rather than as standalone ``tools[]`` entries.
    https://developers.openai.com/api/docs/guides/tools-skills
    """
    skills: list[SkillReferenceParam] = []
    for t in tools:
        if isinstance(t, SkillsToolSchema):
            for s in t.skills:
                entry: SkillReferenceParam = {"type": "skill_reference", "skill_id": s.id}
                # A positive-integer string or "latest"; omitted means the
                # skill's default_version.
                if s.version is not None:
                    entry["version"] = s.version
                skills.append(entry)
    return skills


def merge_skills_into_shell_tools(
    openai_tools: list[dict[str, Any]],
    skills: list[SkillReferenceParam],
) -> list[dict[str, Any]]:
    """Attach skill references to the hosted shell tool's environment.

    Skills require a ``container_auto`` environment. When no shell tool is
    present one is appended (mirrors the Anthropic client auto-adding code
    execution for skills). A ``container_reference`` environment is rejected:
    skills for existing containers are configured at container creation.
    """
    if not skills:
        return openai_tools
    for tool_dict in openai_tools:
        if tool_dict.get("type") == "shell":
            env = tool_dict.setdefault("environment", {"type": "container_auto"})
            if env.get("type") == "container_reference":
                raise ValueError(
                    "SkillsTool cannot be combined with ContainerReferenceEnvironment: "
                    "attach skills when creating the container via ContainerManager instead."
                )
            env["skills"] = skills
            return openai_tools
    openai_tools.append({"type": "shell", "environment": {"type": "container_auto", "skills": skills}})
    return openai_tools


def reject_client_executed_shell(openai_tools: list[dict[str, Any]]) -> None:
    """Refuse a finalized ``shell`` entry the API would run client-side.

    Checked on the finished array, not in :func:`tool_to_responses_api`: the skills path maps
    a bare ``shell`` and :func:`merge_skills_into_shell_tools` gives it ``container_auto`` after.
    """
    for tool_dict in openai_tools:
        if tool_dict.get("type") != "shell":
            continue
        environment = tool_dict.get("environment")
        if environment is None or environment.get("type") == "local":
            raise ClientExecutedShellUnsupportedError()


def responses_api_includes(tools: Iterable[ToolSchema]) -> list[str]:
    includes: list[str] = []
    for t in tools:
        if isinstance(t, WebSearchToolSchema):
            includes.append("web_search_call.action.sources")
        elif isinstance(t, FileSearchToolSchema) and t.include_results:
            # Off by default: results are full text chunks and inflate responses.
            includes.append("file_search_call.results")
    return includes


def normalize_usage(usage: CompletionUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None,
        thinking_tokens=usage.completion_tokens_details.reasoning_tokens if usage.completion_tokens_details else None,
    )


def normalize_responses_usage(usage: ResponseUsage) -> Usage:
    return Usage(
        prompt_tokens=usage.input_tokens,
        completion_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
        thinking_tokens=usage.output_tokens_details.reasoning_tokens,
    )


_MIME_TO_AUDIO_FORMAT: dict[str, Literal["wav", "mp3"]] = {
    "audio/wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
}
"""The only formats Chat Completions accepts for ``input_audio``, keyed by MIME type."""

_CHAT_IMAGE_DETAILS: dict[str, Literal["auto", "low", "high"]] = {"auto": "auto", "low": "low", "high": "high"}

_RESPONSES_IMAGE_DETAILS: dict[str, ImageDetail] = {
    "auto": "auto",
    "low": "low",
    "high": "high",
    "original": "original",
}


def _answered_shell_calls(messages: Sequence[BaseEvent]) -> set[str]:
    """The `call_id`s of hosted shell calls whose output is present in `messages`."""
    return {
        message.item.call_id
        for message in messages
        if isinstance(message, OpenAIServerToolResultEvent) and message.item is not None
    }
