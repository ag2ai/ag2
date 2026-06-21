# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from typing import Any, TypedDict

import boto3
from botocore.config import Config as BotocoreConfig
from fast_depends.library.serializer import SerializerProto
from typing_extensions import Required

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
    Usage,
)
from autogen.beta.response import ResponseProto
from autogen.beta.tools.schemas import ToolSchema

from .mappers import convert_messages, normalize_usage, response_proto_to_output_config, tool_to_api

# End-of-stream sentinel for pulling the sync EventStream via next() without StopIteration
_STREAM_DONE = object()


class CreateOptions(TypedDict, total=False):
    model: Required[str]

    stream: bool
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: list[str]
    additional_model_request_fields: dict[str, Any]
    additional_model_response_field_paths: list[str]
    guardrail_config: dict[str, Any]
    performance_config: dict[str, Any]
    request_metadata: dict[str, str]


class BedrockClient(LLMClient):
    """Amazon Bedrock client for the Converse API (sync boto3 via asyncio.to_thread)."""

    def __init__(
        self,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        profile_name: str | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        botocore_config: Any | None = None,
        session: Any | None = None,
        create_options: CreateOptions | None = None,
    ) -> None:
        self._session = session or boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            profile_name=profile_name,
        )

        config = botocore_config
        if config is None and (timeout is not None or max_retries is not None):
            config_kwargs: dict[str, Any] = {}
            if timeout is not None:
                config_kwargs["connect_timeout"] = timeout
                config_kwargs["read_timeout"] = timeout
            if max_retries is not None:
                config_kwargs["retries"] = {"max_attempts": max_retries, "mode": "standard"}
            config = BotocoreConfig(**config_kwargs)

        self._client_kwargs: dict[str, Any] = {}
        if endpoint_url is not None:
            self._client_kwargs["endpoint_url"] = endpoint_url
        if config is not None:
            self._client_kwargs["config"] = config

        self._client: Any | None = None
        self._create_options = create_options or {}
        self._streaming = self._create_options.get("stream", False)
        self._model: str = self._create_options["model"]

    def _get_client(self) -> Any:
        # Created lazily off the event loop — boto3 loads service models from disk
        if self._client is None:
            self._client = self._session.client("bedrock-runtime", **self._client_kwargs)
        return self._client

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: SerializerProto,
    ) -> ModelResponse:
        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        bedrock_messages = convert_messages(messages, serializer)
        tools_list = [tool_to_api(t) for t in tools]

        kwargs: dict[str, Any] = {
            "modelId": self._model,
            "messages": bedrock_messages,
        }

        system_text = "\n".join(prompt)
        if system_text:
            kwargs["system"] = [{"text": system_text}]

        inference_config: dict[str, Any] = {}
        if (max_tokens := self._create_options.get("max_tokens")) is not None:
            inference_config["maxTokens"] = max_tokens
        if (temperature := self._create_options.get("temperature")) is not None:
            inference_config["temperature"] = temperature
        if (top_p := self._create_options.get("top_p")) is not None:
            inference_config["topP"] = top_p
        if (stop_sequences := self._create_options.get("stop_sequences")) is not None:
            inference_config["stopSequences"] = stop_sequences
        if inference_config:
            kwargs["inferenceConfig"] = inference_config

        # Converse rejects an empty tools list
        if tools_list:
            kwargs["toolConfig"] = {"tools": tools_list}

        if output_config := response_proto_to_output_config(response_schema):
            kwargs["outputConfig"] = output_config

        if (request_fields := self._create_options.get("additional_model_request_fields")) is not None:
            kwargs["additionalModelRequestFields"] = request_fields
        if (response_paths := self._create_options.get("additional_model_response_field_paths")) is not None:
            kwargs["additionalModelResponseFieldPaths"] = response_paths
        if (guardrail := self._create_options.get("guardrail_config")) is not None:
            kwargs["guardrailConfig"] = guardrail
        if (performance := self._create_options.get("performance_config")) is not None:
            kwargs["performanceConfig"] = performance
        if (request_metadata := self._create_options.get("request_metadata")) is not None:
            kwargs["requestMetadata"] = request_metadata

        client = await asyncio.to_thread(self._get_client)

        if self._streaming:
            response = await asyncio.to_thread(client.converse_stream, **kwargs)
            return await self._process_stream(iter(response["stream"]), context)

        response = await asyncio.to_thread(client.converse, **kwargs)
        return await self._process_completion(response, context)

    async def _process_completion(
        self,
        response: dict[str, Any],
        context: "ConversationContext",
    ) -> ModelResponse:
        content_blocks = ((response.get("output") or {}).get("message") or {}).get("content") or []

        text_parts: list[str] = []
        calls: list[ToolCallEvent] = []
        for block in content_blocks:
            if (reasoning := block.get("reasoningContent")) and (
                reasoning_text := (reasoning.get("reasoningText") or {}).get("text")
            ):
                await context.send(ModelReasoning(reasoning_text))

            if text := block.get("text"):
                text_parts.append(text)

            if tool_use := block.get("toolUse"):
                calls.append(
                    ToolCallEvent(
                        id=tool_use["toolUseId"],
                        name=tool_use["name"],
                        arguments=json.dumps(tool_use.get("input") or {}),
                    )
                )

        model_msg: ModelMessage | None = None
        if text_parts:
            model_msg = ModelMessage("".join(text_parts))
            await context.send(model_msg)

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=normalize_usage(response.get("usage") or {}),
            # Converse does not echo the model back — report the configured id
            model=self._model,
            provider="bedrock",
            finish_reason=response.get("stopReason"),
        )

    async def _process_stream(
        self,
        stream: Iterator[dict[str, Any]],
        context: "ConversationContext",
    ) -> ModelResponse:
        full_content: str = ""
        usage = Usage()
        finish_reason: str | None = None
        calls: list[ToolCallEvent] = []

        # toolUse input arrives as partial JSON strings, accumulated by contentBlockIndex
        tool_accs: dict[int, dict[str, str]] = {}

        # Sync EventStream — pull each event off the loop
        while (event := await asyncio.to_thread(next, stream, _STREAM_DONE)) is not _STREAM_DONE:
            if block_start := event.get("contentBlockStart"):
                if tool_use := (block_start.get("start") or {}).get("toolUse"):
                    tool_accs[block_start["contentBlockIndex"]] = {
                        "id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "arguments": "",
                    }

            elif block_delta := event.get("contentBlockDelta"):
                delta = block_delta.get("delta") or {}
                if text := delta.get("text"):
                    full_content += text
                    await context.send(ModelMessageChunk(text))
                if (reasoning := delta.get("reasoningContent")) and (reasoning_text := reasoning.get("text")):
                    await context.send(ModelReasoning(reasoning_text))
                if (tool_use := delta.get("toolUse")) and (
                    acc := tool_accs.get(block_delta["contentBlockIndex"])
                ) is not None:
                    acc["arguments"] += tool_use.get("input") or ""

            elif block_stop := event.get("contentBlockStop"):
                if (acc := tool_accs.pop(block_stop["contentBlockIndex"], None)) is not None:
                    calls.append(ToolCallEvent(id=acc["id"], name=acc["name"], arguments=acc["arguments"] or "{}"))

            elif message_stop := event.get("messageStop"):
                finish_reason = message_stop.get("stopReason")

            elif metadata := event.get("metadata"):
                usage = normalize_usage(metadata.get("usage") or {})

        # Flush accumulators whose contentBlockStop never arrived
        for acc in tool_accs.values():
            calls.append(ToolCallEvent(id=acc["id"], name=acc["name"], arguments=acc["arguments"] or "{}"))

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=self._model,
            provider="bedrock",
            finish_reason=finish_reason,
        )
