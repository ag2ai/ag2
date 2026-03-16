"""Shared budget middleware -- enforces token and call count limits."""

import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from ._state import BUDGET_STATE_KEY, BudgetState
from .base import BaseMiddleware, CallNext
from ._helpers import send_tool_error

logger = logging.getLogger(__name__)


class BudgetExhaustedError(Exception):
    """Raised when a budget limit is exceeded."""


class SharedBudgetMiddleware(BaseMiddleware):
    """
    Enforces shared token and call count budgets across an execution run.

    Budget state is read from ctx.dependencies[BUDGET_STATE_KEY]. A fresh
    BudgetState instance must be injected before each run.

    LLM call short-circuit: when token or LLM call budget is exhausted,
    the middleware emits a ModelResponse denial and returns without calling
    call_next -- the real LLMClient is never invoked.

    Tool call short-circuit: when the tool call budget is exhausted, the
    middleware emits a ToolError and returns without calling call_next.

    Token tracking note: In this PoC, consumed_tokens is NOT auto-incremented
    after an LLM call. The LLMClient emits ModelResponse via ctx.send()
    (fire-and-forget) rather than returning it, so the response is not
    accessible here. Callers can use budget.record_tokens(amount) to
    trigger token exhaustion checks.
    """

    def _get_budget(self, ctx: Any) -> BudgetState:
        """Retrieve and validate the BudgetState from ctx.dependencies.

        Raises TypeError if the key is missing or the value is not a
        BudgetState instance -- fail-closed behaviour.
        """
        budget = ctx.dependencies.get(BUDGET_STATE_KEY)
        if budget is None:
            raise TypeError(
                f"ctx.dependencies[{BUDGET_STATE_KEY!r}] is missing. "
                "Inject a BudgetState before running the agent."
            )
        if not isinstance(budget, BudgetState):
            raise TypeError(
                f"ctx.dependencies[{BUDGET_STATE_KEY!r}] must be a BudgetState, "
                f"got {type(budget).__name__!r}"
            )
        return budget

    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """
        Check budget before forwarding to the next LLM layer.

        Short-circuits (does not call call_next) when token or LLM call
        budget is exhausted.
        """
        from autogen.beta.events import ModelMessage, ModelResponse, ToolCalls

        budget = self._get_budget(ctx)

        allowed, reason = await budget.try_consume_llm_call()
        if not allowed:
            logger.warning(
                "[Budget] LLM call BLOCKED -- %s exhausted "
                "(consumed=%.1f/%.1f tokens, llm_calls=%d/%d)",
                reason,
                budget.consumed_tokens,
                budget.max_tokens,
                budget.llm_calls,
                budget.max_llm_calls,
            )
            await ctx.send(
                ModelResponse(
                    message=ModelMessage(
                        content="[Budget exceeded -- LLM call blocked]"
                    ),
                    tool_calls=ToolCalls(),
                    usage={},
                )
            )
            return

        await call_next(*messages, ctx=ctx, tools=tools)

        logger.info(
            "[Budget] LLM call #%d complete (tokens=%.1f/%.1f)",
            budget.llm_calls,
            budget.consumed_tokens,
            budget.max_tokens,
        )

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """
        Check tool call budget before allowing execution.

        Short-circuits when tool call limit is exhausted.
        """
        budget = self._get_budget(ctx)

        allowed, reason = await budget.try_consume_tool_call()
        if not allowed:
            logger.warning(
                "[Budget] Tool call BLOCKED -- %s exhausted (%d/%d) tool=%s",
                reason,
                budget.tool_calls,
                budget.max_tool_calls,
                getattr(event, "name", "<unknown>"),
            )
            err = BudgetExhaustedError(
                "Tool call budget exhausted"
            )
            await send_tool_error(ctx, event, err)
            return

        await call_next(event, ctx)
        logger.info(
            "[Budget] Tool call #%d complete (tool=%s)",
            budget.tool_calls,
            getattr(event, "name", "<unknown>"),
        )
