# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

import httpx
from anthropic import NOT_GIVEN, AsyncAnthropic
from anthropic.types import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from fast_depends.library.serializer import SerializerProto

from autogen.beta.config.client import LLMClient
from autogen.beta.context import ConversationContext
from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.builtin.code_execution import CodeExecutionToolSchema
from autogen.beta.tools.builtin.skills import SkillsToolSchema
from autogen.beta.tools.schemas import ToolSchema

from .mappers import (
    convert_messages,
    extract_mcp_servers,
    extract_skills_for_container,
    has_file_id_references,
    normalize_usage,
    response_proto_to_output_config,
    tool_to_api,
)


class CreateOptions(TypedDict, total=False):
    model: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
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
        http_client: httpx.AsyncClient | None = None,
        create_options: CreateOptions | None = None,
        prompt_caching: bool = True,
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

        if mcp_servers:
            create_kwargs["extra_headers"] = {"anthropic-beta": "mcp-client-2025-11-20"}
            create_kwargs["extra_body"] = {"mcp_servers": mcp_servers}

        if anthropic_skills:
            create_kwargs["container"] = {"skills": anthropic_skills}
            # Merge beta headers: skills require both code-execution and skills betas
            existing_betas: set[str] = set(
                (create_kwargs.get("extra_headers") or {}).get("anthropic-beta", "").split(",")
            )
            existing_betas.discard("")
            existing_betas.update(["code-execution-2025-08-25", "skills-2025-10-02"])
            create_kwargs.setdefault("extra_headers", {})
            create_kwargs["extra_headers"]["anthropic-beta"] = ",".join(sorted(existing_betas))

        if has_file_id_references(messages):
            existing_betas = set((create_kwargs.get("extra_headers") or {}).get("anthropic-beta", "").split(","))
            existing_betas.discard("")
            existing_betas.add("files-api-2025-04-14")
            create_kwargs.setdefault("extra_headers", {})
            create_kwargs["extra_headers"]["anthropic-beta"] = ",".join(sorted(existing_betas))

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
                anthropic_messages.append({"role": "assistant", "content": response.content})
                create_kwargs["messages"] = anthropic_messages
                response = await self._client.messages.create(**create_kwargs)

            return await self._process_response(response, context)

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
                    )
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

        async for event in stream:
            event_type = getattr(event, "type", None)

            if event_type == "content_block_start":
                block = event.content_block
                if getattr(block, "type", None) == "tool_use":
                    current_tool = {
                        "id": block.id,
                        "name": block.name,
                        "arguments": "",
                    }

            elif event_type == "content_block_delta":
                delta = event.delta
                delta_type = getattr(delta, "type", None)

                if delta_type == "text_delta":
                    full_content += delta.text
                    await context.send(ModelMessageChunk(delta.text))

                elif delta_type == "thinking_delta":
                    await context.send(ModelReasoning(delta.thinking))

                elif delta_type == "input_json_delta" and current_tool is not None:
                    current_tool["arguments"] += delta.partial_json

            elif event_type == "content_block_stop":
                if current_tool is not None:
                    calls.append(
                        ToolCallEvent(
                            id=current_tool["id"],
                            name=current_tool["name"],
                            arguments=current_tool["arguments"],
                        )
                    )
                    current_tool = None

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
