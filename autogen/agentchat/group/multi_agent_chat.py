# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ...doc_utils import export_module
from ...events.agent_events import ErrorEvent, RunCompletionEvent
from ...io.base import IOStream
from ...io.run_response import (
    AsyncRunIterResponse,
    AsyncRunResponse,
    AsyncRunResponseProtocol,
    RunIterResponse,
    RunResponse,
    RunResponseProtocol,
)
from ...io.thread_io_stream import AsyncThreadIOStream, ThreadIOStream
from ...llm_config import LLMConfig
from ..chat import ChatResult
from .context_variables import ContextVariables
from .group_utils import cleanup_temp_user_messages

if TYPE_CHECKING:
    from ...events.base_event import BaseEvent
    from ..agent import Agent
    from .patterns.pattern import Pattern

__all__ = [
    "a_initiate_group_chat",
    "a_run_group_chat",
    "a_run_group_chat_iter",
    "initiate_group_chat",
    "run_group_chat",
    "run_group_chat_iter",
]


@export_module("autogen")
def initiate_group_chat(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> tuple[ChatResult, ContextVariables, "Agent"]:
    """Initialize and run a group chat using a pattern for configuration.

    Args:
        pattern: Pattern object that encapsulates the chat configuration.
        messages: Initial message(s).
        max_rounds: Maximum number of conversation rounds.
        safeguard_policy: Optional safeguard policy dict or path to JSON file.
        safeguard_llm_config: Optional LLM configuration for safeguard checks.
        mask_llm_config: Optional LLM configuration for masking.

    Returns:
        ChatResult:         Conversations chat history.
        ContextVariables:   Updated Context variables.
        "ConversableAgent":   Last speaker.
    """
    # Let the pattern prepare the group chat and all its components
    # Only passing the necessary parameters that aren't already in the pattern
    (
        _,  # agents,
        _,  # wrapped_agents,
        _,  # user_agent,
        context_variables,
        _,  # initial_agent,
        _,  # group_after_work,
        _,  # tool_execution,
        _,  # groupchat,
        manager,
        processed_messages,
        last_agent,
        _,  # group_agent_names,
        _,  # temp_user_list,
    ) = pattern.prepare_group_chat(
        max_rounds=max_rounds,
        messages=messages,
    )

    # Apply safeguards if provided
    if safeguard_policy:
        from .safeguards import apply_safeguard_policy

        apply_safeguard_policy(
            groupchat_manager=manager,
            policy=safeguard_policy,
            safeguard_llm_config=safeguard_llm_config,
            mask_llm_config=mask_llm_config,
        )

    # Start or resume the conversation
    if len(processed_messages) > 1:
        last_agent, last_message = manager.resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    if last_agent is None:
        raise ValueError("No agent selected to start the conversation")

    chat_result = last_agent.initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
        summary_method=pattern.summary_method,
    )

    cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker


@export_module("autogen.agentchat")
async def a_initiate_group_chat(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> tuple[ChatResult, ContextVariables, "Agent"]:
    """Initialize and run a group chat using a pattern for configuration, asynchronously.

    Args:
        pattern: Pattern object that encapsulates the chat configuration.
        messages: Initial message(s).
        max_rounds: Maximum number of conversation rounds.
        safeguard_policy: Optional safeguard policy dict or path to JSON file.
        safeguard_llm_config: Optional LLM configuration for safeguard checks.
        mask_llm_config: Optional LLM configuration for masking.

    Returns:
        ChatResult:         Conversations chat history.
        ContextVariables:   Updated Context variables.
        "ConversableAgent":   Last speaker.
    """
    # Let the pattern prepare the group chat and all its components
    # Only passing the necessary parameters that aren't already in the pattern
    (
        _,  # agents,
        _,  # wrapped_agents,
        _,  # user_agent,
        context_variables,
        _,  # initial_agent,
        _,  # group_after_work,
        _,  # tool_execution,
        _,  # groupchat,
        manager,
        processed_messages,
        last_agent,
        _,  # group_agent_names,
        _,  # temp_user_list,
    ) = pattern.prepare_group_chat(
        max_rounds=max_rounds,
        messages=messages,
    )

    # Apply safeguards if provided
    if safeguard_policy:
        from .safeguards import apply_safeguard_policy

        apply_safeguard_policy(
            groupchat_manager=manager,
            policy=safeguard_policy,
            safeguard_llm_config=safeguard_llm_config,
            mask_llm_config=mask_llm_config,
        )

    # Start or resume the conversation
    if len(processed_messages) > 1:
        last_agent, last_message = await manager.a_resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    if last_agent is None:
        raise ValueError("No agent selected to start the conversation")

    chat_result = await last_agent.a_initiate_chat(
        manager,
        message=last_message,  # type: ignore[arg-type]
        clear_history=clear_history,
        summary_method=pattern.summary_method,
    )

    cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker


@export_module("autogen.agentchat")
def run_group_chat(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> RunResponseProtocol:
    """Run a group chat with multiple agents using the specified pattern.

    This method executes a multi-agent conversation in a background thread and returns
    immediately with a RunResponse object that can be used to iterate over events.

    For step-by-step execution with control over each event, use run_group_chat_iter() instead.

    Args:
        pattern: The pattern that defines how agents interact (e.g., AutoPattern,
            RoundRobinPattern, RandomPattern).
        messages: The initial message(s) to start the conversation. Can be a string
            or a list of message dictionaries.
        max_rounds: Maximum number of conversation rounds. Defaults to 20.
        safeguard_policy: Optional safeguard policy for content filtering.
        safeguard_llm_config: Optional LLM config for safeguard evaluation.
        mask_llm_config: Optional LLM config for content masking.

    Returns:
        RunResponseProtocol
    """
    iostream = ThreadIOStream()
    all_agents = pattern.agents + ([pattern.user_agent] if pattern.user_agent else [])
    response = RunResponse(iostream, agents=all_agents)

    def _initiate_group_chat(
        pattern: "Pattern" = pattern,
        messages: list[dict[str, Any]] | str = messages,
        max_rounds: int = max_rounds,
        safeguard_policy: dict[str, Any] | str | None = safeguard_policy,
        safeguard_llm_config: LLMConfig | None = safeguard_llm_config,
        mask_llm_config: LLMConfig | None = mask_llm_config,
        iostream: ThreadIOStream = iostream,
        response: RunResponse = response,
    ) -> None:
        with IOStream.set_default(iostream):
            try:
                chat_result, context_vars, agent = initiate_group_chat(
                    pattern=pattern,
                    messages=messages,
                    max_rounds=max_rounds,
                    safeguard_policy=safeguard_policy,
                    safeguard_llm_config=safeguard_llm_config,
                    mask_llm_config=mask_llm_config,
                )

                IOStream.get_default().send(
                    RunCompletionEvent(  # type: ignore[call-arg]
                        history=chat_result.chat_history,
                        summary=chat_result.summary,
                        cost=chat_result.cost,
                        last_speaker=agent.name,
                        context_variables=context_vars,
                    )
                )
            except Exception as e:
                response.iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

    threading.Thread(
        target=_initiate_group_chat,
    ).start()

    return response


@export_module("autogen.agentchat")
async def a_run_group_chat(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
) -> AsyncRunResponseProtocol:
    """Async version of run_group_chat for running group chats in async contexts.

    This method executes a multi-agent conversation as an async task and returns
    immediately with an AsyncRunResponse object that can be used to iterate over events.

    For step-by-step execution with control over each event, use a_run_group_chat_iter() instead.

    Args:
        pattern: The pattern that defines how agents interact (e.g., AutoPattern,
            RoundRobinPattern, RandomPattern).
        messages: The initial message(s) to start the conversation. Can be a string
            or a list of message dictionaries.
        max_rounds: Maximum number of conversation rounds. Defaults to 20.
        safeguard_policy: Optional safeguard policy for content filtering.
        safeguard_llm_config: Optional LLM config for safeguard evaluation.
        mask_llm_config: Optional LLM config for content masking.

    Returns:
        AsyncRunResponseProtocol
    """
    iostream = AsyncThreadIOStream()
    all_agents = pattern.agents + ([pattern.user_agent] if pattern.user_agent else [])
    response = AsyncRunResponse(iostream, agents=all_agents)

    async def _initiate_group_chat(
        pattern: "Pattern" = pattern,
        messages: list[dict[str, Any]] | str = messages,
        max_rounds: int = max_rounds,
        safeguard_policy: dict[str, Any] | str | None = safeguard_policy,
        safeguard_llm_config: LLMConfig | None = safeguard_llm_config,
        mask_llm_config: LLMConfig | None = mask_llm_config,
        iostream: AsyncThreadIOStream = iostream,
        response: AsyncRunResponse = response,
    ) -> None:
        with IOStream.set_default(iostream):
            try:
                chat_result, context_vars, agent = await a_initiate_group_chat(
                    pattern=pattern,
                    messages=messages,
                    max_rounds=max_rounds,
                    safeguard_policy=safeguard_policy,
                    safeguard_llm_config=safeguard_llm_config,
                    mask_llm_config=mask_llm_config,
                )

                iostream.send(
                    RunCompletionEvent(  # type: ignore[call-arg]
                        history=chat_result.chat_history,
                        summary=chat_result.summary,
                        cost=chat_result.cost,
                        last_speaker=agent.name,
                        context_variables=context_vars,
                    )
                )
            except Exception as e:
                iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

    task = asyncio.create_task(_initiate_group_chat())
    # prevent the task from being garbage collected
    response._task_ref = task  # type: ignore[attr-defined]
    return response


@export_module("autogen.agentchat")
def run_group_chat_iter(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
    yield_on: Sequence[type["BaseEvent"]] | None = None,
) -> RunIterResponse:
    """Run a group chat with iterator-based stepped execution.

    Iterate over events as they occur. The background thread blocks after each
    event until you advance to the next iteration.

    Example:
        for event in run_group_chat_iter(pattern=pattern, messages="Hello"):
            if isinstance(event, GroupChatRunChatEvent):
                print(f"Speaker: {event.content.speaker}")
            if should_abort(event):
                break  # Cleanup happens automatically

    Args:
        pattern: The pattern that defines how agents interact (e.g., AutoPattern,
            RoundRobinPattern, RandomPattern).
        messages: The initial message(s) to start the conversation. Can be a string
            or a list of message dictionaries.
        max_rounds: Maximum number of conversation rounds. Defaults to 20.
        safeguard_policy: Optional safeguard policy for content filtering.
        safeguard_llm_config: Optional LLM config for safeguard evaluation.
        mask_llm_config: Optional LLM config for content masking.
        yield_on: List of event types to yield. If None, yields all events.
            Common types include TextEvent, ToolCallEvent, GroupChatRunChatEvent,
            and TerminationEvent.

    Returns:
        RunIterResponse: An iterator that yields events as they occur.
    """
    all_agents = pattern.agents + ([pattern.user_agent] if pattern.user_agent else [])

    def create_thread(iostream: ThreadIOStream) -> threading.Thread:
        def _initiate_group_chat() -> None:
            with IOStream.set_default(iostream):
                try:
                    chat_result, context_vars, agent = initiate_group_chat(
                        pattern=pattern,
                        messages=messages,
                        max_rounds=max_rounds,
                        safeguard_policy=safeguard_policy,
                        safeguard_llm_config=safeguard_llm_config,
                        mask_llm_config=mask_llm_config,
                    )

                    IOStream.get_default().send(
                        RunCompletionEvent(  # type: ignore[call-arg]
                            history=chat_result.chat_history,
                            summary=chat_result.summary,
                            cost=chat_result.cost,
                            last_speaker=agent.name,
                            context_variables=context_vars,
                        )
                    )
                except Exception as e:
                    iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

        return threading.Thread(target=_initiate_group_chat, daemon=True)

    return RunIterResponse(
        start_thread_func=create_thread,
        yield_on=yield_on,
        agents=all_agents,
    )


@export_module("autogen.agentchat")
def a_run_group_chat_iter(
    pattern: "Pattern",
    messages: list[dict[str, Any]] | str,
    max_rounds: int = 20,
    safeguard_policy: dict[str, Any] | str | None = None,
    safeguard_llm_config: LLMConfig | None = None,
    mask_llm_config: LLMConfig | None = None,
    yield_on: Sequence[type["BaseEvent"]] | None = None,
) -> AsyncRunIterResponse:
    """Async version of run_group_chat_iter for async contexts.

    Iterate over events as they occur using async for. The background thread blocks
    after each event until you advance to the next iteration.

    Example:
        async for event in a_run_group_chat_iter(pattern=pattern, messages="Hello"):
            if isinstance(event, GroupChatRunChatEvent):
                print(f"Speaker: {event.content.speaker}")
            if should_abort(event):
                break  # Cleanup happens automatically

    Args:
        pattern: The pattern that defines how agents interact (e.g., AutoPattern,
            RoundRobinPattern, RandomPattern).
        messages: The initial message(s) to start the conversation. Can be a string
            or a list of message dictionaries.
        max_rounds: Maximum number of conversation rounds. Defaults to 20.
        safeguard_policy: Optional safeguard policy for content filtering.
        safeguard_llm_config: Optional LLM config for safeguard evaluation.
        mask_llm_config: Optional LLM config for content masking.
        yield_on: List of event types to yield. If None, yields all events.
            Common types include TextEvent, ToolCallEvent, GroupChatRunChatEvent,
            and TerminationEvent.

    Returns:
        AsyncRunIterResponse: An async iterator that yields events as they occur.
    """
    all_agents = pattern.agents + ([pattern.user_agent] if pattern.user_agent else [])

    def create_thread(iostream: ThreadIOStream) -> threading.Thread:
        async def _async_initiate_group_chat() -> None:
            chat_result, context_vars, agent = await a_initiate_group_chat(
                pattern=pattern,
                messages=messages,
                max_rounds=max_rounds,
                safeguard_policy=safeguard_policy,
                safeguard_llm_config=safeguard_llm_config,
                mask_llm_config=mask_llm_config,
            )

            iostream.send(
                RunCompletionEvent(  # type: ignore[call-arg]
                    history=chat_result.chat_history,
                    summary=chat_result.summary,
                    cost=chat_result.cost,
                    last_speaker=agent.name,
                    context_variables=context_vars,
                )
            )

        def _run_in_thread() -> None:
            with IOStream.set_default(iostream):
                try:
                    asyncio.run(_async_initiate_group_chat())
                except Exception as e:
                    iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

        return threading.Thread(target=_run_in_thread, daemon=True)

    return AsyncRunIterResponse(
        start_thread_func=create_thread,
        yield_on=yield_on,
        agents=all_agents,
    )
