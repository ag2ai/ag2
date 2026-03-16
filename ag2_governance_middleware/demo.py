#!/usr/bin/env python3
"""
AG2 Governance Middleware -- Demo

Verifies 4 success conditions without API keys using mock LLM and mock Context:

  1. Budget LLM block   -- max_llm_calls=0 blocks the LLM call
  2. Budget tool block  -- max_tool_calls=0 blocks the tool call
  3. Policy deny        -- shell_exec in denied_tools emits ToolError
  4. Secret redaction   -- credit card number is replaced before LLM sees content
"""

import asyncio
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Any

# -- Logging ---------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s -- %(message)s",
)

# -- Path setup ------------------------------------------------------------
# ag2_governance_middleware/ lives inside D:\work\Projects\ag2\
# autogen.beta lives in D:\work\Projects\ag2\autogen\beta\
# We add the ag2 root so both are importable.

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AG2_ROOT = os.path.dirname(_THIS_DIR)
if _AG2_ROOT not in sys.path:
    sys.path.insert(0, _AG2_ROOT)

# -- Import autogen.beta event types --------------------------------------

from autogen.beta.events import (  # noqa: E402
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolError,
)

# -- Mock context ---------------------------------------------------------
# Real Context requires fast_depends (Provider). We use a minimal mock that
# only exposes the attributes the middleware accesses: dependencies and send().


@dataclass
class MockContext:
    """Minimal stand-in for autogen.beta.context.Context."""

    sent_events: list[BaseEvent] = field(default_factory=list)
    dependencies: dict[Any, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    prompt: list[str] = field(default_factory=list)

    async def send(self, event: BaseEvent) -> None:
        self.sent_events.append(event)


# -- Import middleware -----------------------------------------------------

from ag2_governance_middleware import (  # noqa: E402
    BUDGET_STATE_KEY,
    BudgetState,
    PolicyDenyMiddleware,
    SecretRedactionMiddleware,
    SharedBudgetMiddleware,
)
from ag2_governance_middleware.base import build_middleware_client  # noqa: E402

# -- Test helpers ---------------------------------------------------------


async def run_test(name: str, coro: "asyncio.Coroutine[Any, Any, bool]") -> bool:
    """Run a single test coroutine and print PASS/FAIL."""
    print(f"\n--- {name} ---")
    try:
        result = await coro
        label = "PASS" if result else "FAIL"
        print(f"[{label}] {name}")
        return bool(result)
    except Exception as exc:
        print(f"[FAIL] {name} -- exception: {exc!r}")
        return False


# -- Test 1: Budget LLM block ---------------------------------------------


async def test_budget_llm_block() -> bool:
    """max_llm_calls=0 must block the LLM call without invoking call_next."""
    call_next_invoked = False

    async def real_client(*messages: BaseEvent, ctx: Any, tools: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True  # Must NOT be reached.

    mw = SharedBudgetMiddleware()
    client = build_middleware_client(real_client, [mw])

    state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=0)
    ctx = MockContext(dependencies={BUDGET_STATE_KEY: state})

    await client(ModelRequest(content="hello"), ctx=ctx, tools=[])

    denial_emitted = any(
        isinstance(e, ModelResponse)
        and e.message is not None
        and "Budget exceeded" in e.message.content
        for e in ctx.sent_events
    )
    print(
        f"  call_next_invoked={call_next_invoked}, "
        f"blocked_llm_calls={state.blocked_llm_calls}, "
        f"denial_emitted={denial_emitted}"
    )
    return (not call_next_invoked) and denial_emitted


# -- Test 2: Budget tool block --------------------------------------------


async def test_budget_tool_block() -> bool:
    """max_tool_calls=0 must block the tool call and emit ToolError."""
    mw = SharedBudgetMiddleware()

    state = BudgetState(max_tokens=10_000, max_tool_calls=0, max_llm_calls=10)
    ctx = MockContext(dependencies={BUDGET_STATE_KEY: state})

    tool_call = ToolCall(id="tc-1", name="web_search", arguments='{"query": "test"}')

    call_next_invoked = False

    async def next_handler(event: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, tool_call, ctx)

    error_emitted = any(
        isinstance(e, ToolError) and e.name == "web_search" for e in ctx.sent_events
    )
    print(
        f"  call_next_invoked={call_next_invoked}, "
        f"blocked_tool_calls={state.blocked_tool_calls}, "
        f"ToolError_emitted={error_emitted}"
    )
    return (not call_next_invoked) and error_emitted


# -- Test 3: Policy deny --------------------------------------------------


async def test_policy_deny() -> bool:
    """Tool in denied_tools must produce ToolError without calling next."""
    mw = PolicyDenyMiddleware(denied_tools={"shell_exec"})

    ctx = MockContext()
    tool_call = ToolCall(id="tc-2", name="shell_exec", arguments='{"cmd": "ls"}')

    call_next_invoked = False

    async def next_handler(event: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, tool_call, ctx)

    error_emitted = any(
        isinstance(e, ToolError) and e.name == "shell_exec" for e in ctx.sent_events
    )
    print(
        f"  call_next_invoked={call_next_invoked}, "
        f"ToolError_emitted={error_emitted}"
    )
    return (not call_next_invoked) and error_emitted


# -- Test 4: Secret redaction ---------------------------------------------

_CREDIT_CARD_PATTERN = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
_TEST_SECRET = "4111 1111 1111 1111"


async def test_redaction() -> bool:
    """Credit card number must be redacted before call_next receives content."""
    mw = SecretRedactionMiddleware(patterns=[_CREDIT_CARD_PATTERN])

    received_content: list[str] = []

    async def capture_client(*messages: BaseEvent, ctx: Any, tools: Any) -> None:
        for msg in messages:
            if isinstance(msg, ModelRequest):
                received_content.append(msg.content)

    client = build_middleware_client(capture_client, [mw])
    ctx = MockContext()

    await client(
        ModelRequest(content=f"My card is {_TEST_SECRET}"),
        ctx=ctx,
        tools=[],
    )

    if not received_content:
        print("  ERROR: call_next was never invoked")
        return False

    content = received_content[0]
    secret_absent = _TEST_SECRET not in content
    placeholder_present = "[REDACTED]" in content
    print(
        f"  received_content={content!r}, "
        f"secret_absent={secret_absent}, "
        f"placeholder_present={placeholder_present}"
    )
    return secret_absent and placeholder_present


# -- Main -----------------------------------------------------------------


async def main() -> None:
    tests = [
        ("Budget LLM block", test_budget_llm_block),
        ("Budget tool block", test_budget_tool_block),
        ("Policy deny", test_policy_deny),
        ("Redaction", test_redaction),
    ]
    results: list[bool] = []
    for name, test_fn in tests:
        result = await run_test(name, test_fn())
        results.append(result)

    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("ALL PASS")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
