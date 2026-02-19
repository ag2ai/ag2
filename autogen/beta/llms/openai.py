from typing import Any, Literal

from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from autogen.beta.events import (
    BaseEvent,
    ModelReasoning,
    ModelRequest,
    ModelResponse,
    StreamModelResult,
    ToolResult,
    UserMessage,
)
from autogen.beta.llms.client import LLMClient
from autogen.beta.stream import Stream

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        reasoning_effort: ReasoningEffort | None = None,
        **kwargs: Any,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self._model = model
        self._reasoning_effort = reasoning_effort

    async def __call__(self, *messages: BaseEvent, stream: Stream) -> None:
        openai_messages = self._convert_messages(messages)

        should_stream = True

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": openai_messages,
            "stream": should_stream,
        }
        if self._reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = self._reasoning_effort

        response = await self._client.chat.completions.create(**create_kwargs)

        if should_stream:
            await self._process_stream(response, stream)
        else:
            await self._process_completion(response, stream)

    def _convert_messages(self, messages: tuple[BaseEvent, ...]) -> list[dict[str, str]]:
        result: list[dict[str, str]] = []

        for message in messages:
            if isinstance(message, UserMessage):
                result.append({"role": "user", "content": message.content})
            elif isinstance(message, ModelRequest):
                result.append({"role": "user", "content": message.prompt})
            elif isinstance(message, ModelResponse):
                result.append({"role": "assistant", "content": message.response})
            elif isinstance(message, ToolResult):
                result.append({
                    "role": "tool",
                    "tool_call_id": message.name,
                    "content": message.result,
                })

        return result

    async def _process_completion(
        self,
        completion: ChatCompletion,
        stream: Stream,
    ) -> None:
        for choice in completion.choices:
            msg = choice.message

            if r := getattr(msg, "reasoning", None):
                await stream.send(ModelReasoning(reasoning=r))

            if c := msg.content:
                await stream.send(ModelResponse(content=c))

    async def _process_stream(
        self,
        response_stream: AsyncStream[ChatCompletionChunk],
        stream: Stream,
    ) -> None:
        full_content: str = ""

        async for chunk in response_stream:
            for choice in chunk.choices:
                delta = choice.delta

                if r := getattr(delta, "reasoning_content", None):
                    await stream.send(ModelReasoning(reasoning=r))

                if c := delta.content:
                    full_content += c
                    await stream.send(StreamModelResult(content=c))

        if full_content:
            await stream.send(ModelResponse(content=full_content))

        #     delta = chunk.choices[0].delta
        #
        #     if delta.content:
        #         content = delta.content
        #         full_content += content
        #         await stream.send(StreamModelResult(result=content))
        #
        #     if delta.tool_calls:
        #         for tool_call in delta.tool_calls:
        #             if tool_call.function:
        #                 if current_tool_call is None:
        #                     current_tool_call = {
        #                         "name": tool_call.function.name or "",
        #                         "arguments": "",
        #                     }
        #                 if tool_call.function.arguments:
        #                     current_tool_call["arguments"] += tool_call.function.arguments
        #                     await stream.send(StreamToolCall(
        #                         name=current_tool_call["name"],
        #                         arguments=tool_call.function.arguments,
        #                     ))
        #
        # if current_tool_call and current_tool_call["name"]:
        #     await stream.send(ToolCall(
        #         name=current_tool_call["name"],
        #         arguments=current_tool_call["arguments"],
        #     ))
