# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Core step() function - a single complete agent turn with automatic tool execution."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.events.base_event import BaseEvent
from autogen.events.client_events import StreamEvent
from autogen.io import IOStream

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call made during a step."""

    name: str
    arguments: dict[str, Any]
    result: str
    is_success: bool
    call_id: str | None = None


@dataclass
class UsageRecord:
    """Accumulated token usage from one or more LLM calls during a step."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    model: str = ""


@dataclass
class StepResult:
    """Result of a single agent step.

    Attributes:
        content: The final text response from the agent.
        messages: The full message history including system, user, tool calls, and final response.
        tool_calls_made: Record of all tool calls executed during this step.
        usage: Accumulated token usage and cost across all LLM calls in this step.
    """

    content: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: list[ToolCallRecord] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)


class _QuietIOStream:
    """IOStream that suppresses all output.

    Used during step() to prevent AG2's internal events (ExecuteFunctionEvent,
    ExecutedFunctionEvent, etc.) from leaking to the terminal and flashing
    over a Live dashboard.
    """

    def print(self, *objects: Any, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
        pass

    def send(self, message: BaseEvent) -> None:
        pass


class _CallbackIOStream(_QuietIOStream):
    """IOStream that intercepts StreamEvent and forwards text deltas to a callback.

    Inherits _QuietIOStream so all non-StreamEvent output is suppressed.
    """

    def __init__(self, callback: Callable[[str], None]) -> None:
        self._callback = callback

    def send(self, message: BaseEvent) -> None:
        if isinstance(message, StreamEvent):
            # @wrap_event wraps StreamEvent so message.content is the inner
            # event object, not the raw text string.  Unwrap one level.
            inner = message.content
            text = inner.content if hasattr(inner, "content") else str(inner)
            self._callback(text)


def _normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Normalize a tool call dict to ensure it has the right structure for the API."""
    result = {"type": "function"}
    if "id" in tool_call:
        result["id"] = tool_call["id"]
    if "function" in tool_call:
        result["function"] = tool_call["function"]
    return result


async def step(
    agent: ConversableAgent,
    messages: list[dict[str, Any]] | str,
    *,
    max_turns: int = 20,
    verbose: bool = False,
    stream: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> StepResult:
    """Execute one complete agent turn with automatic tool execution.

    This is the core building block for team orchestration. It sends messages
    to an agent's LLM, automatically executes any tool calls the LLM makes,
    and loops until the LLM produces a final text response.

    Args:
        agent: A ConversableAgent with an LLM config and optionally registered tools.
        messages: Either a string (treated as a user message) or a list of message dicts.
            If a list, these are appended after the agent's system message.
        max_turns: Maximum number of LLM call rounds (to prevent infinite tool loops).
        verbose: Whether to log detailed information about the step.
        stream: Whether to enable streaming from the LLM client.
        on_token: Callback invoked with each text delta when streaming. Called from the
            executor thread, so it must be thread-safe.

    Returns:
        StepResult with the final text content, full message history, and tool call records.

    Raises:
        ValueError: If the agent has no LLM client configured.
        RuntimeError: If max_turns is exceeded without a final text response.
    """
    if agent.client is None:
        raise ValueError(f"Agent '{agent.name}' has no LLM client configured. Set llm_config.")

    # Normalize messages input
    conversation = [{"role": "user", "content": messages}] if isinstance(messages, str) else list(messages)

    tool_calls_made: list[ToolCallRecord] = []
    usage = UsageRecord()

    # OpenAIWrapper.create() merges kwargs with the stored config list via
    # {**config, **client_config}, so client_config's stream=False (a Pydantic
    # default in LLMConfigEntry) overrides our runtime stream=True.  Remove
    # the stored default so our value survives the merge.
    if stream and hasattr(agent, "client") and hasattr(agent.client, "_config_list"):
        for cfg in agent.client._config_list:
            cfg.pop("stream", None)

    return await _step_loop(
        agent,
        conversation,
        tool_calls_made,
        usage,
        max_turns=max_turns,
        verbose=verbose,
        stream=stream,
        on_token=on_token,
    )


async def _step_loop(
    agent: ConversableAgent,
    conversation: list[dict[str, Any]],
    tool_calls_made: list[ToolCallRecord],
    usage: UsageRecord,
    *,
    max_turns: int,
    verbose: bool,
    stream: bool,
    on_token: Callable[[str], None] | None,
) -> StepResult:
    """Inner loop for step(), wrapped in a quiet IOStream.

    Separated so the ``with IOStream.set_default(...)`` block can cleanly
    cover the entire turn loop including tool execution, preventing AG2's
    internal events (ExecuteFunctionEvent, etc.) from leaking to the terminal.
    """
    # Suppress AG2's internal IOStream output for the entire step so it
    # doesn't flash over a Live dashboard.  The executor thread sets its
    # own IOStream for streaming tokens; this only covers the async
    # context (tool execution, extract_text, etc.).
    with IOStream.set_default(_QuietIOStream()):
        for turn in range(max_turns):
            # Build full message list: system + conversation
            full_messages = list(agent._oai_system_message) + conversation

            # Call the LLM (run sync create in executor to not block)
            create_kwargs: dict[str, Any] = {
                "messages": full_messages,
                "cache": agent.client_cache,
            }
            if stream:
                create_kwargs["stream"] = True

            def _run_create() -> Any:
                if stream and on_token is not None:
                    # Install a callback IOStream so StreamEvent tokens reach on_token
                    cb_stream = _CallbackIOStream(on_token)
                    with IOStream.set_default(cb_stream):
                        return agent.client.create(**create_kwargs)
                return agent.client.create(**create_kwargs)

            response = await asyncio.get_event_loop().run_in_executor(None, _run_create)

            # Accumulate token usage from this LLM call
            if hasattr(response, "usage") and response.usage is not None:
                usage.prompt_tokens += response.usage.prompt_tokens or 0
                usage.completion_tokens += response.usage.completion_tokens or 0
                usage.total_tokens += (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
            if hasattr(response, "cost"):
                usage.cost += response.cost or 0
            if hasattr(response, "model") and response.model:
                usage.model = response.model

            # Extract the response - convert to dict if it's a message object
            extracted = agent.client.extract_text_or_completion_object(response)[0]

            if not isinstance(extracted, (str, dict)) and hasattr(extracted, "model_dump"):
                extracted = extracted.model_dump()

            # Handle string response (no tool calls) - we're done
            if isinstance(extracted, str):
                conversation.append({"role": "assistant", "content": extracted})
                return StepResult(
                    content=extracted,
                    messages=conversation,
                    tool_calls_made=tool_calls_made,
                    usage=usage,
                )

            # Handle dict response - may contain tool_calls
            if isinstance(extracted, dict):
                tool_calls = extracted.get("tool_calls")
                content = extracted.get("content")

                if not tool_calls:
                    # No tool calls, just content.
                    # Responses API may return content as a list of blocks
                    # e.g. [{"type": "output_text", "text": "..."}]
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("text"):
                                text_parts.append(block["text"])
                        final_content = "\n".join(text_parts) if text_parts else ""
                    else:
                        final_content = content or ""
                    conversation.append({"role": "assistant", "content": final_content})
                    return StepResult(
                        content=final_content,
                        messages=conversation,
                        tool_calls_made=tool_calls_made,
                        usage=usage,
                    )

                # Has tool calls - add assistant message then execute each tool
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [_normalize_tool_call(tc) for tc in tool_calls],
                }
                conversation.append(assistant_msg)

                for tool_call in tool_calls:
                    func_info = tool_call.get("function", {})
                    func_name = func_info.get("name", "")
                    func_args = func_info.get("arguments", "{}")
                    call_id = tool_call.get("id")

                    if verbose:
                        logger.info(f"Executing tool: {func_name}({func_args})")

                    # Execute the tool
                    is_success, result_dict = await agent.a_execute_function(
                        {"name": func_name, "arguments": func_args},
                        call_id=call_id,
                    )

                    result_content = str(result_dict.get("content", ""))

                    # Record the tool call
                    try:
                        parsed_args = json.loads(func_args)
                    except (json.JSONDecodeError, TypeError):
                        parsed_args = {"_raw": func_args}

                    tool_calls_made.append(
                        ToolCallRecord(
                            name=func_name,
                            arguments=parsed_args,
                            result=result_content,
                            is_success=is_success,
                            call_id=call_id,
                        )
                    )

                    # Add tool response to conversation
                    tool_response: dict[str, Any] = {
                        "role": "tool",
                        "content": result_content,
                    }
                    if call_id:
                        tool_response["tool_call_id"] = call_id
                    conversation.append(tool_response)

                # Loop back to call LLM again with tool results
                continue

            # Unexpected response type
            raise RuntimeError(f"Unexpected LLM response type: {type(extracted)}")

    raise RuntimeError(
        f"Agent '{agent.name}' exceeded max_turns={max_turns} without producing a final response. "
        "This usually means the agent is stuck in a tool execution loop."
    )
