# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Assembler — composable context assembly for LLM calls.

The assembler composes AssemblyPolicy instances into a middleware that
transforms (prompts, events) before each LLM call.

Policies compose left-to-right: each sees the output of the previous.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from autogen.beta.context import ConversationContext as Context
from autogen.beta.events import BaseEvent, ModelResponse
from autogen.beta.middleware.base import BaseMiddleware, LLMCall


@runtime_checkable
class AssemblyPolicy(Protocol):
    """Transforms context before each LLM invocation.

    A policy receives (prompts, events) and returns modified (prompts, events).
    Policies compose: each sees the output of the previous.

    Policies are pure transforms with one exception: they may read from
    KnowledgeStore or Hub (via context.dependencies). They must not have
    side effects on the stream.
    """

    name: str

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        """Transform prompts and events. Return modified copies."""
        ...


class AssemblerMiddleware(BaseMiddleware):
    """Runs assembly policies before each LLM call.

    Sits at the outermost position in the middleware chain. Runs all
    policies in order, transforming (prompts, events) before they
    reach the LLM client.

    Middleware ordering in Agent._execute()::

        1. AssemblerMiddleware(policies)     -- outermost: assembles context
        2. AlertPolicy (in assembly chain)   -- injects observer alerts
        3. CompactionMiddleware              -- triggers compaction after turns
        4. AggregationMiddleware             -- triggers aggregation after turns
        5. User-provided middleware           -- logging, retry, etc.
        6. LLM client call                    -- innermost: sends to model
    """

    def __init__(
        self,
        event: BaseEvent,
        context: Context,
        *,
        policies: list[AssemblyPolicy],
    ) -> None:
        super().__init__(event, context)
        self._policies = policies

    async def on_llm_call(
        self,
        call_next: LLMCall,
        events: Sequence[BaseEvent],
        context: Context,
    ) -> ModelResponse:
        prompts = list(context.prompt)
        event_list = list(events)

        for policy in self._policies:
            prompts, event_list = await policy.apply(prompts, event_list, context)

        # Temporarily replace prompts, restore in finally
        original_prompt = context.prompt
        context.prompt = prompts
        try:
            return await call_next(event_list, context)
        finally:
            context.prompt = original_prompt

    @staticmethod
    def validate_order(policies: list[AssemblyPolicy]) -> list[str]:
        """Check for known-problematic policy orderings. Returns warnings."""
        warnings: list[str] = []
        names = [p.name for p in policies]

        # Reduction policies should come AFTER injection policies
        # because injections add context that reduction should then trim.
        reduction = {"sliding_window", "token_budget"}
        injection = {"episodic_memory", "working_memory", "topic_inbox", "alert"}

        for i, name in enumerate(names):
            if name in reduction:
                for j in range(i + 1, len(names)):
                    if names[j] in injection:
                        warnings.append(
                            f"Policy '{name}' (index {i}) runs before '{names[j]}' (index {j}). "
                            f"Injection policies should generally run before reduction policies "
                            f"so injected context is included in the reduction budget."
                        )
        return warnings
