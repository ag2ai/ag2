from collections.abc import Iterable
from typing import Any, Literal

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelMessageChunk,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolResults,
)
from autogen.beta.stream import Context
from autogen.beta.tools import Tool

from .client import LLMClient

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        streaming: bool = False,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._streaming = streaming

    async def __call__(
        self,
        *messages: BaseEvent,
        ctx: Context,
        tools: Iterable[Tool],
    ) -> None:
        openai_messages = self._convert_messages(ctx.prompt, messages)

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "stream": self._streaming,
        }
        if self._reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = self._reasoning_effort

        response = await self._client.chat.completions.create(
            **create_kwargs,
            tools=[t.schema.to_api() for t in tools],
        )

        if self._streaming:
            await self._process_stream(response, ctx)
        else:
            await self._process_completion(response, ctx)

    def _convert_messages(
        self,
        system_prompt: Iterable[str],
        messages: tuple[BaseEvent, ...],
    ) -> list[dict[str, str]]:
        result: list[dict[str, str]] = [{"content": p, "role": "developer"} for p in system_prompt]

        for message in messages:
            if isinstance(message, ModelRequest):
                result.append({
                    "role": "user",
                    "content": message.content,
                })
            elif isinstance(message, ModelResponse):
                msg = {
                    "role": "assistant",
                    "content": message.message.content if message.message else None,
                }
                tool_calls = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "arguments": c.arguments,
                            "name": c.name,
                        },
                    }
                    for c in message.tool_calls.calls
                ]
                if tool_calls:
                    msg["tool_calls"] = tool_calls
                result.append(msg)
            elif isinstance(message, ToolResults):
                for r in message.results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": r.id,
                        "content": r.content,
                    })

        return result

    async def _process_completion(
        self,
        completion: ChatCompletion,
        ctx: Context,
    ) -> ToolCalls | ModelMessage:
        for choice in completion.choices:
            msg = choice.message

            if r := getattr(msg, "reasoning", None):
                await ctx.send(ModelReasoning(content=r))

            model_msg: ModelMessage | None = None
            if c := msg.content:
                model_msg = ModelMessage(content=c)
                await ctx.send(model_msg)

            calls = [
                ToolCall(
                    id=c.id,
                    name=c.function.name,
                    arguments=c.function.arguments,
                )
                for c in (msg.tool_calls or ())
            ]

            await ctx.send(
                ModelResponse(
                    message=model_msg,
                    tool_calls=ToolCalls(calls=calls),
                )
            )

    async def _process_stream(
        self,
        response_stream: AsyncStream[ChatCompletionChunk],
        ctx: Context,
    ) -> ToolCalls | ModelMessage:
        full_content: str = ""

        calls = []
        async for chunk in response_stream:
            for choice in chunk.choices:
                delta = choice.delta

                if r := getattr(delta, "reasoning_content", None):
                    await ctx.send(ModelReasoning(content=r))

                if c := delta.content:
                    full_content += c
                    await ctx.send(ModelMessageChunk(content=c))

                for c in delta.tool_calls or []:
                    calls.append(
                        ToolCall(
                            id=c.id,
                            name=c.function.name,
                            arguments=c.function.arguments,
                        )
                    )

        message: ModelMessage | None = None
        if full_content:
            message = ModelMessage(content=full_content)
            await ctx.send(message)

        await ctx.send(
            ModelResponse(
                message=message,
                tool_calls=ToolCalls(calls=calls),
            )
        )
