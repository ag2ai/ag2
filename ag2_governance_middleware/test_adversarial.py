"""
Adversarial tests for AG2 Governance Middleware PoC.

Tests attack vectors: TOCTOU races, None/corrupted inputs, boundary
conditions, predicate failures, and full-stack integration.

Run with:
    python -m pytest ag2_governance_middleware/test_adversarial.py -v
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from autogen.beta.events import (
    BaseEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolCalls,
    ToolError,
)

from ag2_governance_middleware import (
    BUDGET_STATE_KEY,
    BudgetState,
    PolicyDenyMiddleware,
    SecretRedactionMiddleware,
    SharedBudgetMiddleware,
    build_middleware_client,
    build_middleware_tool_chain,
)


# -- Mock Context -------------------------------------------------------------


@dataclass
class MockContext:
    sent_events: list[BaseEvent] = field(default_factory=list)
    dependencies: dict[Any, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    prompt: list[str] = field(default_factory=list)

    async def send(self, event: BaseEvent) -> None:
        self.sent_events.append(event)


# -- Mock ToolCall with controllable attributes --------------------------------


@dataclass
class FakeToolCall:
    """Allows setting name/arguments to arbitrary values including None."""

    id: str = "tc-adv"
    name: Any = "test_tool"
    arguments: Any = "{}"


# =============================================================================
# Category 1: Corrupted Input
# =============================================================================


@pytest.mark.asyncio
async def test_policy_event_name_none() -> None:
    """event.name=None must be denied (fail-closed), not bypass deny list."""
    mw = PolicyDenyMiddleware(denied_tools={"shell_exec"})
    ctx = MockContext()
    event = FakeToolCall(name=None, arguments="{}")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    assert not call_next_invoked, "call_next must not be invoked for None name"
    assert any(isinstance(e, ToolError) for e in ctx.sent_events), "ToolError must be emitted"


@pytest.mark.asyncio
async def test_policy_event_arguments_none() -> None:
    """event.arguments=None must not crash json.loads (TypeError caught)."""
    mw = PolicyDenyMiddleware(
        denied_predicates=[lambda name, args: args.get("danger", False)]
    )
    ctx = MockContext()
    event = FakeToolCall(name="safe_tool", arguments=None)

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    # Should proceed: predicate sees empty args dict, returns False
    assert call_next_invoked, "call_next must be invoked when predicate does not deny"


@pytest.mark.asyncio
async def test_policy_event_arguments_binary() -> None:
    """Binary (non-JSON) arguments must be DENIED (fail-closed)."""
    mw = PolicyDenyMiddleware(
        denied_predicates=[lambda name, args: False]
    )
    ctx = MockContext()
    event = FakeToolCall(name="tool", arguments=b"\x00\xff\xfe")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)
    # Fail-closed: unparseable arguments -> DENY, call_next NOT invoked
    assert not call_next_invoked, "call_next must NOT be invoked for unparseable binary args"
    assert len(ctx.sent_events) > 0, "ToolError must be sent for denied binary args"


@pytest.mark.asyncio
async def test_redaction_content_none() -> None:
    """ModelRequest with content=None must not crash redaction middleware."""
    mw = SecretRedactionMiddleware(patterns=[re.compile(r"SECRET")])

    received: list[Any] = []

    async def capture(*messages: Any, ctx: Any, tools: Any) -> None:
        received.extend(messages)

    client = build_middleware_client(capture, [mw])

    event = ModelRequest(content="no secret here")
    object.__setattr__(event, "content", None)  # Force None after construction

    ctx = MockContext()
    await client(event, ctx=ctx, tools=[])

    assert len(received) == 1, "Event must be forwarded (no crash on None content)"


@pytest.mark.asyncio
async def test_redaction_content_non_string() -> None:
    """ModelRequest with non-str content (int) must pass through unchanged."""
    mw = SecretRedactionMiddleware(patterns=[re.compile(r"\d+")])

    received: list[Any] = []

    async def capture(*messages: Any, ctx: Any, tools: Any) -> None:
        received.extend(messages)

    client = build_middleware_client(capture, [mw])

    event = ModelRequest(content="hello")
    object.__setattr__(event, "content", 12345)  # Force int after construction

    ctx = MockContext()
    await client(event, ctx=ctx, tools=[])

    assert len(received) == 1, "Event must be forwarded"
    assert received[0].content == 12345, "Non-str content must pass through unchanged"


# =============================================================================
# Category 2: Predicate Failure (fail-closed)
# =============================================================================


@pytest.mark.asyncio
async def test_policy_predicate_exception_denies() -> None:
    """Predicate that raises must result in DENY (fail-closed)."""

    def exploding_predicate(name: str, args: dict) -> bool:
        raise RuntimeError("I broke")

    mw = PolicyDenyMiddleware(denied_predicates=[exploding_predicate])
    ctx = MockContext()
    event = ToolCall(id="tc-x", name="innocent_tool", arguments="{}")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    assert not call_next_invoked, "call_next must not be invoked when predicate raises"
    assert any(isinstance(e, ToolError) for e in ctx.sent_events), "ToolError must be emitted"


@pytest.mark.asyncio
async def test_policy_first_predicate_ok_second_explodes() -> None:
    """First predicate passes, second raises -- must still DENY."""

    def ok_predicate(name: str, args: dict) -> bool:
        return False  # Allow

    def bad_predicate(name: str, args: dict) -> bool:
        raise ValueError("boom")

    mw = PolicyDenyMiddleware(denied_predicates=[ok_predicate, bad_predicate])
    ctx = MockContext()
    event = ToolCall(id="tc-y", name="tool_y", arguments="{}")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    assert not call_next_invoked, "call_next must not be invoked when second predicate raises"
    assert any(isinstance(e, ToolError) for e in ctx.sent_events), "ToolError must be emitted"


# =============================================================================
# Category 3: Boundary Conditions (N-1, N, N+1)
# =============================================================================


@pytest.mark.asyncio
async def test_budget_llm_exactly_at_limit() -> None:
    """LLM call at exact limit (calls=max) must be blocked."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=2)

    allowed1, _ = await state.try_consume_llm_call()
    assert allowed1, "1st call must be allowed"
    allowed2, _ = await state.try_consume_llm_call()
    assert allowed2, "2nd call must be allowed"

    # 3rd must be blocked
    allowed3, reason = await state.try_consume_llm_call()
    assert not allowed3, "3rd call must be blocked when max_llm_calls=2"
    assert reason == "llm_calls", f"Expected reason='llm_calls', got {reason!r}"
    assert state.blocked_llm_calls == 1, "blocked_llm_calls must be 1"


@pytest.mark.asyncio
async def test_budget_tool_boundary_n_minus_1() -> None:
    """Tool call at N-1 must succeed, at N must succeed, at N+1 must block."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=3, max_llm_calls=10)

    results = []
    for _ in range(5):
        allowed, _reason = await state.try_consume_tool_call()
        results.append(allowed)

    expected = [True, True, True, False, False]
    assert results == expected, f"Expected {expected}, got {results}"


@pytest.mark.asyncio
async def test_budget_token_exhaustion_blocks_llm() -> None:
    """consumed_tokens >= max_tokens must block LLM calls even if llm_calls < max."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    # Use internal setter since direct assignment is guarded
    state._unsafe_set("consumed_tokens", 100.0)

    allowed, reason = await state.try_consume_llm_call()
    assert not allowed, "LLM call must be blocked when tokens exhausted"
    assert reason == "tokens", f"Expected reason='tokens', got {reason!r}"


@pytest.mark.asyncio
async def test_budget_token_just_under_limit() -> None:
    """consumed_tokens just under max must still allow LLM call."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    state._unsafe_set("consumed_tokens", 99.9)

    allowed, _ = await state.try_consume_llm_call()
    assert allowed, "LLM call must be allowed when tokens not yet exhausted"


@pytest.mark.asyncio
async def test_budget_record_tokens_rejects_nan() -> None:
    """record_tokens with NaN must raise ValueError (NaN would bypass token_exhausted)."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    with pytest.raises(ValueError, match="finite"):
        await state.record_tokens(float("nan"))


@pytest.mark.asyncio
async def test_budget_record_tokens_rejects_negative() -> None:
    """record_tokens with negative amount must raise ValueError."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    with pytest.raises(ValueError, match="non-negative"):
        await state.record_tokens(-1.0)


@pytest.mark.asyncio
async def test_budget_state_counter_mutation_guard() -> None:
    """Direct assignment to counter fields must raise AttributeError after init."""
    state = BudgetState(max_tokens=100.0, max_tool_calls=10, max_llm_calls=10)
    with pytest.raises(AttributeError, match="read-only"):
        state.consumed_tokens = 999.9  # type: ignore[misc]


@pytest.mark.asyncio
async def test_budget_state_rejects_nan_max_tokens() -> None:
    """BudgetState must reject NaN for max_tokens (would silently bypass token_exhausted)."""
    with pytest.raises(ValueError, match="max_tokens"):
        BudgetState(max_tokens=float("nan"), max_tool_calls=10, max_llm_calls=10)


@pytest.mark.asyncio
async def test_budget_state_rejects_inf_max_tokens() -> None:
    """BudgetState must reject Inf for max_tokens."""
    with pytest.raises(ValueError, match="max_tokens"):
        BudgetState(max_tokens=float("inf"), max_tool_calls=10, max_llm_calls=10)


@pytest.mark.asyncio
async def test_budget_state_rejects_negative_max_llm_calls() -> None:
    """BudgetState must reject negative max_llm_calls."""
    with pytest.raises(ValueError, match="max_llm_calls"):
        BudgetState(max_tokens=1000.0, max_tool_calls=10, max_llm_calls=-1)


# =============================================================================
# Category 4: Concurrent Access (TOCTOU)
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_llm_budget_consumption() -> None:
    """Multiple concurrent try_consume_llm_call must not exceed max."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=5)

    async def consume() -> bool:
        allowed, _ = await state.try_consume_llm_call()
        return allowed

    results = await asyncio.gather(*[consume() for _ in range(20)])

    successes = sum(1 for r in results if r)
    assert successes == 5, f"Expected exactly 5 successes, got {successes}"
    assert state.llm_calls == 5, f"Expected llm_calls=5, got {state.llm_calls}"


@pytest.mark.asyncio
async def test_concurrent_tool_budget_consumption() -> None:
    """Multiple concurrent try_consume_tool_call must not exceed max."""
    state = BudgetState(max_tokens=10_000, max_tool_calls=3, max_llm_calls=10)

    async def consume() -> tuple[bool, str]:
        return await state.try_consume_tool_call()

    results = await asyncio.gather(*[consume() for _ in range(15)])

    successes = sum(1 for allowed, _reason in results if allowed)
    assert successes == 3, f"Expected exactly 3 successes, got {successes}"
    assert state.tool_calls == 3, f"Expected tool_calls=3, got {state.tool_calls}"


# =============================================================================
# Category 5: Security
# =============================================================================


@pytest.mark.asyncio
async def test_tool_error_content_does_not_leak_tool_name() -> None:
    """ToolError.content must not include the tool name (info leak prevention)."""
    mw = PolicyDenyMiddleware(denied_tools={"secret_internal_tool"})
    ctx = MockContext()
    event = ToolCall(id="tc-sec", name="secret_internal_tool", arguments="{}")

    async def next_handler(ev: Any, context: Any) -> None:
        pass

    await mw.on_tool_call(next_handler, event, ctx)

    tool_errors = [e for e in ctx.sent_events if isinstance(e, ToolError)]
    assert tool_errors, "ToolError must be emitted"
    for err_event in tool_errors:
        assert "secret_internal_tool" not in err_event.content, (
            f"Tool name must not appear in ToolError.content, got: {err_event.content!r}"
        )


@pytest.mark.asyncio
async def test_budget_injectable_key_type_rejected() -> None:
    """ctx.dependencies with wrong type for BUDGET_STATE_KEY must raise TypeError."""
    mw = SharedBudgetMiddleware()
    ctx = MockContext(dependencies={BUDGET_STATE_KEY: "not-a-budget-state"})

    with pytest.raises(TypeError, match="BudgetState"):
        await mw.on_llm_call(
            lambda *a, **kw: asyncio.sleep(0),
            ModelRequest(content="test"),
            ctx=ctx,
            tools=[],
        )


@pytest.mark.asyncio
async def test_budget_missing_key_raises_type_error() -> None:
    """ctx.dependencies without BUDGET_STATE_KEY must raise TypeError."""
    mw = SharedBudgetMiddleware()
    ctx = MockContext(dependencies={})

    with pytest.raises(TypeError, match="missing"):
        await mw.on_llm_call(
            lambda *a, **kw: asyncio.sleep(0),
            ModelRequest(content="test"),
            ctx=ctx,
            tools=[],
        )


@pytest.mark.asyncio
async def test_policy_deny_list_not_mutated_after_init() -> None:
    """Mutating the original set after construction must not affect _denied_tools."""
    original_set = {"tool_a", "tool_b"}
    mw = PolicyDenyMiddleware(denied_tools=original_set)

    # Mutate the original
    original_set.add("tool_c")

    ctx = MockContext()
    event = ToolCall(id="tc-mut", name="tool_c", arguments="{}")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    # tool_c was added AFTER construction -- must not be denied
    assert call_next_invoked, "tool_c must not be denied (copy taken at init)"


# =============================================================================
# Category 6: Full-Stack Integration
# =============================================================================


@pytest.mark.asyncio
async def test_full_stack_redaction_then_policy_then_budget() -> None:
    """
    Full middleware stack: Redaction -> Policy -> Budget.
    Secret in tool arguments: redaction applies to both LLM and tool calls.
    """
    redaction_mw = SecretRedactionMiddleware(
        patterns=[re.compile(r"sk-[a-zA-Z0-9]+")]
    )
    policy_mw = PolicyDenyMiddleware(denied_tools={"dangerous_tool"})
    budget_state = BudgetState(max_tokens=10_000, max_tool_calls=10, max_llm_calls=10)

    stack = [redaction_mw, policy_mw, SharedBudgetMiddleware()]
    ctx = MockContext(dependencies={BUDGET_STATE_KEY: budget_state})

    # Test 1: LLM call with secret -- should be redacted
    received_content: list[str] = []

    async def capture_llm(*messages: Any, ctx: Any, tools: Any) -> None:
        for msg in messages:
            if isinstance(msg, ModelRequest):
                received_content.append(msg.content)

    llm_client = build_middleware_client(capture_llm, stack)
    await llm_client(
        ModelRequest(content="Key is sk-abc123xyz"),
        ctx=ctx,
        tools=[],
    )

    assert received_content, "LLM call must reach capture_llm"
    assert "sk-abc123xyz" not in received_content[0], (
        f"Secret must be redacted, got: {received_content[0]!r}"
    )

    # Test 2: Tool call to denied tool -- should be blocked by policy
    tool_chain = build_middleware_tool_chain(
        lambda ev, ctx: asyncio.sleep(0),
        stack,
    )

    await tool_chain(
        ToolCall(id="tc-fs", name="dangerous_tool", arguments="{}"),
        ctx,
    )

    policy_blocked = any(
        isinstance(e, ToolError) and e.name == "dangerous_tool"
        for e in ctx.sent_events
    )
    assert policy_blocked, "dangerous_tool must be blocked by policy"

    # Test 3: Allowed tool call -- should consume budget
    await tool_chain(
        ToolCall(id="tc-ok", name="safe_tool", arguments="{}"),
        ctx,
    )

    assert budget_state.tool_calls == 1, (
        f"Only safe_tool should consume budget, got tool_calls={budget_state.tool_calls}"
    )


@pytest.mark.asyncio
async def test_policy_deny_list_and_predicate_combined() -> None:
    """Tool in deny list is blocked even if predicate would allow it."""
    mw = PolicyDenyMiddleware(
        denied_tools={"blocked_tool"},
        denied_predicates=[lambda name, args: False],  # Would allow
    )
    ctx = MockContext()
    event = ToolCall(id="tc-combo", name="blocked_tool", arguments="{}")

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    assert not call_next_invoked, "call_next must not be invoked for denied tool"
    assert any(isinstance(e, ToolError) for e in ctx.sent_events), "ToolError must be emitted"


@pytest.mark.asyncio
async def test_redaction_oversized_content_passes_through() -> None:
    """Content exceeding max_content_bytes must pass through without redaction (ReDoS guard)."""
    mw = SecretRedactionMiddleware(
        patterns=[re.compile(r"SECRET")],
        max_content_bytes=10,
    )

    received: list[Any] = []

    async def capture(*messages: Any, ctx: Any, tools: Any) -> None:
        received.extend(messages)

    client = build_middleware_client(capture, [mw])
    oversized = "SECRET " * 10  # 70 chars, exceeds 10 byte limit
    event = ModelRequest(content=oversized)

    ctx = MockContext()
    await client(event, ctx=ctx, tools=[])

    assert len(received) == 1, "Oversized event must still be forwarded"
    assert received[0].content == oversized, "Oversized content must pass through unchanged"


@pytest.mark.asyncio
async def test_policy_non_dict_json_args_denied() -> None:
    """Non-dict JSON arguments (array) must be treated as DENY (fail-closed)."""
    mw = PolicyDenyMiddleware(
        denied_predicates=[lambda name, args: False]  # Would allow a dict
    )
    ctx = MockContext()
    # JSON array is valid JSON but not a dict
    event = ToolCall(id="tc-arr", name="some_tool", arguments='["not", "a", "dict"]')

    call_next_invoked = False

    async def next_handler(ev: Any, context: Any) -> None:
        nonlocal call_next_invoked
        call_next_invoked = True

    await mw.on_tool_call(next_handler, event, ctx)

    assert not call_next_invoked, "Non-dict JSON args must be denied (fail-closed)"
    assert any(isinstance(e, ToolError) for e in ctx.sent_events), "ToolError must be emitted"
