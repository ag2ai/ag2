# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

import httpx2
from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic.types import (
    ContainerUploadBlock,
    Message,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from fast_depends.library.serializer import SerializerProto

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
)
from ag2.response import ResponseProto
from ag2.tools.builtin.code_execution import CodeExecutionToolSchema
from ag2.tools.builtin.skills import SkillsToolSchema
from ag2.tools.schemas import ToolSchema

from .events import (
    AnthropicContainerUploadEvent,
    AnthropicRedactedThinkingEvent,
    AnthropicServerToolCallEvent,
    AnthropicServerToolResultBlockType,
    AnthropicServerToolResultEvent,
)

logger = logging.getLogger(__name__)


def _tool_use_vendor_metadata(block: ToolUseBlock) -> dict[str, Any]:
    """Provider fields ``ToolCallEvent`` has no column for but must replay."""
    meta: dict[str, Any] = {}
    if (caller := getattr(block, "caller", None)) is not None:
        meta["caller"] = caller.model_dump(mode="json") if hasattr(caller, "model_dump") else caller
    if (toolset := getattr(block, "toolset_name", None)) is not None:
        meta["toolset_name"] = toolset
    return meta


from .mappers import (
    convert_messages,
    extract_mcp_servers,
    extract_skills_for_container,
    has_file_id_references,
    merge_sampling_into_extra_body,
    normalize_usage,
    response_proto_to_output_config,
    take_sampling_fields,
    tool_to_api,
)


class CreateOptions(TypedDict, total=False):
    # `temperature`, `top_p` and `top_k` are absent on purpose: `anthropic` 1.x
    # removed them from the Messages API signature, so passing one here is a
    # `TypeError`. `AnthropicConfig` forwards them in the request's extra body
    # instead — see `AnthropicConfig._request_extra_body`.
    model: str
    max_tokens: int
    stop_sequences: list[str] | None
    stream: bool
    metadata: dict[str, str] | None
    service_tier: str | None


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        create_options: CreateOptions | None = None,
        prompt_caching: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout if timeout is not None else NOT_GIVEN,
            max_retries=max_retries,
            default_headers=default_headers,
            http_client=http_client,
        )

        self._create_options = {k: v for k, v in (create_options or {}).items() if k != "stream"}
        self._streaming = (create_options or {}).get("stream", False)
        self._prompt_caching = prompt_caching

        # A caller reaching this class directly (rather than through
        # `AnthropicConfig`) may still be passing the sampling parameters 1.x
        # removed from the method signature. Route them the same way, instead of
        # letting them reach the SDK as a bare `TypeError` at request time.
        self._extra_body = merge_sampling_into_extra_body(take_sampling_fields(self._create_options), extra_body)

        self._default_betas = _default_betas(default_headers)

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        anthropic_messages = convert_messages(messages, serializer)

        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        system: Any = (
            self._build_system(prompt)
            if context.prompt or (response_schema and response_schema.system_prompt)
            else NOT_GIVEN
        )

        if self._prompt_caching and anthropic_messages:
            self._inject_cache_control(anthropic_messages)

        tools_schemas = list(tools)
        tools_without_skills = [t for t in tools_schemas if not isinstance(t, SkillsToolSchema)]
        anthropic_skills = extract_skills_for_container(tools_schemas)

        if anthropic_skills and not any(isinstance(t, CodeExecutionToolSchema) for t in tools_without_skills):
            tools_without_skills.append(CodeExecutionToolSchema())

        tools_list = [tool_to_api(t) for t in tools_without_skills]
        mcp_servers = extract_mcp_servers(tools_without_skills)

        kwargs: dict[str, Any] = {}
        if r := response_proto_to_output_config(response_schema):
            kwargs["output_config"] = r

        create_kwargs: dict[str, Any] = {
            **self._create_options,
            **kwargs,
            "system": system,
            "messages": anthropic_messages,
            "tools": tools_list if tools_list else NOT_GIVEN,
        }

        betas: set[str] = set()

        if mcp_servers:
            betas.add("mcp-client-2025-11-20")
            create_kwargs["extra_body"] = {"mcp_servers": mcp_servers}

        if anthropic_skills:
            create_kwargs["container"] = {"skills": anthropic_skills}
            # Skills require both the code-execution and the skills beta.
            betas.update(("code-execution-2025-08-25", "skills-2025-10-02"))

        if has_file_id_references(messages):
            betas.add("files-api-2025-04-14")

        if betas:
            # `anthropic` 1.x matches header names case-insensitively, so this
            # per-request header *replaces* a same-named default rather than being
            # sent alongside it. Fold in the betas the user set as defaults, or
            # asking for one of ours would silently drop theirs.
            create_kwargs["extra_headers"] = {"anthropic-beta": ",".join(sorted(betas | self._default_betas))}

        # User keys win over framework-set keys (e.g. mcp_servers) on collision.
        if self._extra_body:
            existing = create_kwargs.get("extra_body") or {}
            create_kwargs["extra_body"] = {**existing, **self._extra_body}

        max_continuations = 5

        if self._streaming:
            async with self._client.messages.stream(**create_kwargs) as stream:
                result = await self._process_stream(stream, context)
                final_msg = await stream.get_final_message()

            for _ in range(max_continuations):
                if result.finish_reason != "pause_turn":
                    break
                anthropic_messages.append({"role": "assistant", "content": final_msg.content})
                create_kwargs["messages"] = anthropic_messages
                async with self._client.messages.stream(**create_kwargs) as stream:
                    result = await self._process_stream(stream, context)
                    final_msg = await stream.get_final_message()

            return result
        else:
            response = await self._client.messages.create(**create_kwargs)

            for _ in range(max_continuations):
                if response.stop_reason != "pause_turn":
                    break
                await self._emit_builtin_tool_events(response.content, context)
                anthropic_messages.append({"role": "assistant", "content": response.content})
                create_kwargs["messages"] = anthropic_messages
                response = await self._client.messages.create(**create_kwargs)

            return await self._process_response(response, context)

    async def _emit_builtin_tool_events(
        self,
        content_blocks: list[Any],
        context: "ConversationContext",
    ) -> None:
        """Emit typed server-tool events for server-side tool blocks."""
        for block in content_blocks:
            if isinstance(block, ServerToolUseBlock):
                if call_event := AnthropicServerToolCallEvent.from_block(block):
                    await context.send(call_event)
            elif isinstance(block, AnthropicServerToolResultBlockType) and (
                result_event := AnthropicServerToolResultEvent.from_block(block)
            ):
                await context.send(result_event)

    def _build_system(self, prompt: Iterable[str]) -> Any:
        text = "\n".join(prompt)
        if self._prompt_caching:
            return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return text

    @staticmethod
    def _inject_cache_control(messages: list[dict[str, Any]]) -> None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
                elif isinstance(content, list) and content:
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                break

    async def _process_response(
        self,
        response: Message,
        context: "ConversationContext",
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []

        for block in response.content:
            if isinstance(block, ThinkingBlock):
                if block.thinking:
                    await context.send(ModelReasoning(block.thinking))

            elif isinstance(block, TextBlock):
                model_msg = ModelMessage(block.text)
                await context.send(model_msg)

            elif isinstance(block, ToolUseBlock):
                calls.append(
                    ToolCallEvent(
                        id=block.id,
                        name=block.name,
                        arguments=json.dumps(block.input),
                        vendor_metadata=_tool_use_vendor_metadata(block),
                    )
                )

            elif isinstance(block, ServerToolUseBlock):
                if call_event := AnthropicServerToolCallEvent.from_block(block):
                    await context.send(call_event)

            elif isinstance(block, RedactedThinkingBlock):
                await context.send(AnthropicRedactedThinkingEvent(block=block))

            elif isinstance(block, ContainerUploadBlock):
                await context.send(AnthropicContainerUploadEvent(block=block))

            elif isinstance(block, AnthropicServerToolResultBlockType) and (
                result_event := AnthropicServerToolResultEvent.from_block(block)
            ):
                await context.send(result_event)

            else:
                # Without this the next block type Anthropic adds disappears with
                # no event, no warning and no trace in history.
                logger.warning(
                    "Dropping unhandled Anthropic content block type=%r (%s)",
                    getattr(block, "type", None),
                    type(block).__name__,
                )

        usage = normalize_usage(response.usage.model_dump() if response.usage else {})

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=response.model,
            provider="anthropic",
            finish_reason=response.stop_reason,
        )

    async def _process_stream(
        self,
        stream: Any,
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        calls: list[ToolCallEvent] = []

        current_tool: dict[str, Any] | None = None
        current_server: dict[str, Any] | None = None

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_start":
                block = event.content_block
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    current_tool = {
                        "id": block.id,
                        "name": block.name,
                        "arguments": "",
                        "vendor_metadata": _tool_use_vendor_metadata(block),
                    }
                elif block_type == "server_tool_use":
                    # ``input`` is empty here and arrives as input_json_delta; the
                    # event is built at content_block_stop so it carries the whole
                    # thing. Building it now replays an empty input.
                    current_server = {"block": block, "arguments": ""}
                elif isinstance(block, RedactedThinkingBlock):
                    await context.send(AnthropicRedactedThinkingEvent(block=block))
                elif isinstance(block, ContainerUploadBlock):
                    await context.send(AnthropicContainerUploadEvent(block=block))
                elif isinstance(block, AnthropicServerToolResultBlockType):
                    if result_event := AnthropicServerToolResultEvent.from_block(block):
                        await context.send(result_event)
                elif block_type not in ("text", "thinking"):
                    logger.warning(
                        "Dropping unhandled Anthropic content block type=%r (%s)",
                        block_type,
                        type(block).__name__,
                    )

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)

                if delta_type == "text_delta":
                    full_content += delta.text
                    await context.send(ModelMessageChunk(delta.text))

                elif delta_type == "thinking_delta":
                    await context.send(ModelReasoning(delta.thinking))

                elif delta_type == "input_json_delta":
                    if current_tool is not None:
                        current_tool["arguments"] += delta.partial_json
                    elif current_server is not None:
                        current_server["arguments"] += delta.partial_json

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    calls.append(
                        ToolCallEvent(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=current_tool["arguments"],
                            vendor_metadata=current_tool["vendor_metadata"],
                        )
                    )
                    current_tool = None
                elif current_server is not None:
                    block = current_server["block"]
                    raw = current_server["arguments"]
                    try:
                        # ``max_tokens`` can cut a block mid-JSON; keep the block
                        # rather than fail the turn on a partial input.
                        block = block.model_copy(update={"input": json.loads(raw)}) if raw else block
                    except json.JSONDecodeError:
                        logger.warning(
                            "Incomplete server_tool_use input for id=%s; keeping %r",
                            block.id,
                            block.input,
                        )
                    if call_event := AnthropicServerToolCallEvent.from_block(block):
                        await context.send(call_event)
                    current_server = None

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        final_message = await stream.get_final_message()
        # Mapped to our usage keys
        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(final_message.usage.model_dump() if final_message.usage else {}),
            model=final_message.model,
            provider="anthropic",
            finish_reason=final_message.stop_reason,
        )


def _default_betas(default_headers: dict[str, str] | None) -> frozenset[str]:
    """The ``anthropic-beta`` values a caller set as client defaults.

    Matched case-insensitively, and last-one-wins between two spellings of the
    same name, because that is what the SDK's own header merge does with them.
    """
    if not default_headers:
        return frozenset()
    value = ""
    for name, header in default_headers.items():
        if name.lower() == "anthropic-beta":
            value = header
    return frozenset(part.strip() for part in value.split(",") if part.strip())
