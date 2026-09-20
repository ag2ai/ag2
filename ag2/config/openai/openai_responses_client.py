# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Iterable, Sequence
from itertools import chain
from typing import Any, TypedDict

import httpx2
from fast_depends.library.serializer import SerializerProto
from openai import DEFAULT_MAX_RETRIES, AsyncOpenAI, AsyncStream, Omit, not_given, omit
from openai.types import ChatModel
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseShellCallCommandDeltaEvent,
    ResponseShellCallOutputContentDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_create_params import PromptCacheOptions
from openai.types.responses.response_output_item import ImageGenerationCall
from typing_extensions import Required

from ag2.config.client import LLMClient
from ag2.context import ConversationContext
from ag2.events import (
    BaseEvent,
    BinaryResult,
    ModelMessage,
    ModelMessageChunk,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    Usage,
)
from ag2.response import ResponseProto
from ag2.tools.builtin.skills import SkillsToolSchema
from ag2.tools.schemas import ToolSchema

from .events import (
    OpenAIPromptCacheDiagnostics,
    OpenAIReasoningEvent,
    OpenAIServerToolCallEvent,
    OpenAIServerToolResultEvent,
    OpenAIShellCommandChunk,
    OpenAIShellOutputChunk,
    ShellCallTracker,
)
from .mappers import (
    events_to_responses_input,
    extract_skills_for_shell,
    merge_skills_into_shell_tools,
    normalize_responses_usage,
    reject_client_executed_shell,
    response_proto_to_text_config,
    responses_api_includes,
    tool_to_responses_api,
)


class CreateOptions(TypedDict, total=False):
    model: Required[ChatModel | str]

    temperature: float | None | Omit
    top_p: float | None | Omit
    max_output_tokens: int | None | Omit
    max_tool_calls: int | None | Omit
    parallel_tool_calls: bool
    top_logprobs: int | None | Omit
    store: bool | None
    metadata: dict[str, str] | None | Omit
    prompt_cache_key: str | Omit
    prompt_cache_options: PromptCacheOptions | Omit
    service_tier: str | None | Omit
    user: str
    stream: bool
    truncation: str | None | Omit


class OpenAIResponsesClient(LLMClient):
    def __init__(
        self,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | None = None,
        websocket_base_url: str | None = None,
        timeout: Any = not_given,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: dict[str, str] | None = None,
        default_query: dict[str, object] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        create_options: CreateOptions | None = None,
        prompt_cache_diagnostics: bool = False,
    ) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
        )

        self._create_options = create_options or {}
        self._streaming = self._create_options.get("stream", False)
        self._prompt_cache_diagnostics = prompt_cache_diagnostics
        self._last_response_id: str | None = None

    def _request_options(self) -> dict[str, Any]:
        """The create options for this call, diagnosing against the last response if asked to.

        Only the comparison id varies between calls; everything else is fixed at construction.
        Returns a plain mapping rather than ``CreateOptions`` because that TypedDict declares
        ``service_tier`` and ``truncation`` as ``str`` where the SDK wants a ``Literal``;
        spreading it as a typed mapping fails on those fields, which are not this change's.
        """
        options = dict(self._create_options)
        if not self._prompt_cache_diagnostics or self._last_response_id is None:
            return options

        configured = self._create_options.get("prompt_cache_options")
        if configured and "comparison_response_id" in configured:
            # The caller named a response to compare against; that choice outranks the chain.
            return options

        chained: PromptCacheOptions = {"comparison_response_id": self._last_response_id}
        options["prompt_cache_options"] = {**configured, **chained} if configured else chained
        return options

    @staticmethod
    def _comparison_id(options: dict[str, Any]) -> str | None:
        """The response this request asked to be compared against, if it asked at all."""
        configured: PromptCacheOptions | None = options.get("prompt_cache_options") or None
        return configured.get("comparison_response_id") if configured else None

    async def _report_cache_diagnostics(
        self,
        response: Response,
        context: "ConversationContext",
        comparison_id: str | None,
    ) -> None:
        if diagnostics := OpenAIPromptCacheDiagnostics.from_response(response, requested_comparison_id=comparison_id):
            await context.send(diagnostics)

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable[ToolSchema],
        response_schema: ResponseProto | None,
        serializer: "SerializerProto",
    ) -> ModelResponse:
        input_items = events_to_responses_input(messages, serializer)

        if response_schema and response_schema.system_prompt:
            prompt: Iterable[str] = chain(context.prompt, (response_schema.system_prompt,))
        else:
            prompt = context.prompt

        instructions = "\n".join(prompt) or None

        tools_list = list(tools)
        openai_skills = extract_skills_for_shell(tools_list)
        tools_list = [t for t in tools_list if not isinstance(t, SkillsToolSchema)]
        openai_tools = merge_skills_into_shell_tools(
            [tool_to_responses_api(t) for t in tools_list],
            openai_skills,
        )
        reject_client_executed_shell(openai_tools)

        kwargs: dict[str, Any] = {}
        if r := response_proto_to_text_config(response_schema):
            kwargs["text"] = r

        includes = responses_api_includes(tools_list)
        if self._create_options.get("store") is False:
            # Stateless mode: the API persists nothing, so replaying a reasoning item
            # by id on a later turn (a tool loop always does) will fail. Asking
            # for the encrypted payload makes those items self-contained.
            includes.append("reasoning.encrypted_content")
        if includes:
            kwargs["include"] = includes

        options = self._request_options()
        response = await self._client.responses.create(
            **options,
            **kwargs,
            input=input_items,
            instructions=instructions,
            tools=openai_tools or omit,
        )

        comparison_id = self._comparison_id(options)
        if self._streaming:
            result = await self._process_stream(response, context, comparison_id=comparison_id)
        else:
            result = await self._process_response(response, context, comparison_id=comparison_id)

        # A turn that reported no id leaves the previous one standing: a stale comparison
        # still names a real response, where `None` would silently stop diagnosing.
        if result.response_id:
            self._last_response_id = result.response_id
        return result

    async def _process_response(
        self,
        response: Response,
        context: "ConversationContext",
        *,
        comparison_id: str | None = None,
    ) -> ModelResponse:
        model_msg: ModelMessage | None = None
        calls: list[ToolCallEvent] = []
        files: list[BinaryResult] = []
        shell_calls = ShellCallTracker()

        for item in response.output:
            if isinstance(item, ResponseReasoningItem):
                summaries = [s.text for s in item.summary if s.text]
                if not summaries:
                    await context.send(OpenAIReasoningEvent("", item=item))
                else:
                    for text in summaries:
                        await context.send(OpenAIReasoningEvent(text, item=item))

            elif isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if hasattr(part, "text") and part.text:
                        model_msg = ModelMessage(part.text)
                        await context.send(model_msg)

            elif isinstance(item, ResponseFunctionToolCall):
                calls.append(
                    ToolCallEvent(
                        id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                    )
                )

            elif call_event := OpenAIServerToolCallEvent.from_item(item):
                await context.send(call_event)
                if isinstance(item, ResponseFunctionShellToolCall):
                    shell_calls.opened(item, event_id=call_event.id)
                result_event = OpenAIServerToolResultEvent.from_item(item, parent_id=call_event.id)
                if result_event:
                    await context.send(result_event)
                    if isinstance(item, ImageGenerationCall) and item.result:
                        binary = result_event.result.parts[0]
                        files.append(BinaryResult(binary.data, metadata=result_event.result.metadata))

            elif isinstance(item, ResponseFunctionShellToolCallOutput):
                if shell_result := shell_calls.close(item):
                    await context.send(shell_result)

        usage = normalize_responses_usage(response.usage) if response.usage else Usage()

        await self._report_cache_diagnostics(response, context, comparison_id)

        return ModelResponse(
            message=model_msg,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=response.model,
            provider="openai",
            finish_reason=response.status,
            response_id=response.id,
            files=files,
        )

    async def _process_stream(
        self,
        response_stream: AsyncStream[ResponseStreamEvent],
        context: "ConversationContext",
        *,
        comparison_id: str | None = None,
    ) -> ModelResponse:
        full_content: str = ""
        calls: list[ToolCallEvent] = []
        files: list[BinaryResult] = []
        finish_reason: str | None = None
        resolved_model: str | None = None
        response_id: str | None = None
        usage = Usage()
        shell_calls = ShellCallTracker()

        async for event in response_stream:
            if isinstance(event, ResponseTextDeltaEvent):
                full_content += event.delta
                await context.send(ModelMessageChunk(event.delta))

            elif isinstance(event, ResponseShellCallCommandDeltaEvent):
                await context.send(
                    OpenAIShellCommandChunk(
                        event.delta,
                        command_index=event.command_index,
                        output_index=event.output_index,
                    )
                )

            elif isinstance(event, ResponseShellCallOutputContentDeltaEvent):
                await context.send(
                    OpenAIShellOutputChunk(
                        command_index=event.command_index,
                        output_index=event.output_index,
                        item_id=event.item_id,
                        stdout=event.delta.stdout,
                        stderr=event.delta.stderr,
                    )
                )

            elif isinstance(event, ResponseOutputItemDoneEvent):
                # Builtin and reasoning events are emitted on Done so the typed
                # SDK object carried by the event is fully populated (Added fires
                # before the server-side tool has executed — code/outputs missing).

                if isinstance(event.item, ResponseReasoningItem):
                    summaries = [s.text for s in event.item.summary if s.text]
                    if not summaries:
                        await context.send(OpenAIReasoningEvent("", item=event.item))
                    else:
                        for text in summaries:
                            await context.send(OpenAIReasoningEvent(text, item=event.item))

                elif isinstance(event.item, ResponseFunctionToolCall):
                    calls.append(
                        ToolCallEvent(
                            id=event.item.call_id,
                            name=event.item.name,
                            arguments=event.item.arguments,
                        )
                    )

                elif call_event := OpenAIServerToolCallEvent.from_item(event.item):
                    await context.send(call_event)
                    if isinstance(event.item, ResponseFunctionShellToolCall):
                        shell_calls.opened(event.item, event_id=call_event.id)
                    result_event = OpenAIServerToolResultEvent.from_item(event.item, parent_id=call_event.id)
                    if result_event:
                        await context.send(result_event)
                        if isinstance(event.item, ImageGenerationCall) and event.item.result:
                            binary = result_event.result.parts[0]
                            files.append(BinaryResult(binary.data, metadata=result_event.result.metadata))

                elif isinstance(event.item, ResponseFunctionShellToolCallOutput):
                    if shell_result := shell_calls.close(event.item):
                        await context.send(shell_result)

            elif isinstance(event, ResponseCompletedEvent):
                # Stream finished
                if event.response.usage:
                    usage = normalize_responses_usage(event.response.usage)

                finish_reason = event.response.status
                resolved_model = event.response.model
                response_id = event.response.id

                await self._report_cache_diagnostics(event.response, context, comparison_id)

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(full_content)
            await context.send(message)

        return ModelResponse(
            message=message,
            tool_calls=ToolCallsEvent(calls),
            usage=usage,
            model=resolved_model,
            provider="openai",
            finish_reason=finish_reason,
            response_id=response_id,
            files=files,
        )
