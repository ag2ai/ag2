"""Budget state shared across a single execution run."""

import asyncio
import math
from dataclasses import dataclass, field

# Sentinel key -- uses a versioned string to be pickle-safe while remaining
# unlikely to collide with user-defined keys in context.dependencies.
BUDGET_STATE_KEY: str = "__veronica_budget_state_v1"


@dataclass
class BudgetState:
    """
    Execution-scoped budget counters.

    Create a fresh instance per agent run and inject via:
        ctx.dependencies[BUDGET_STATE_KEY] = BudgetState(...)

    All mutation of counter fields must go through the provided async methods
    (try_consume_llm_call, try_consume_tool_call, record_tokens) which are
    protected by _lock (asyncio.Lock).

    Direct assignment to counter fields (consumed_tokens, llm_calls, etc.)
    is NOT lock-protected. External code should use record_tokens() to update
    consumed_tokens safely.

    asyncio.Lock is bound to the event loop at first acquisition. Create
    BudgetState instances within the async context where they will be used,
    or use asyncio.run_coroutine_threadsafe() to avoid loop-binding issues.
    Requires Python 3.10+ (asyncio.Lock no longer requires a running loop at
    construction time since 3.10).
    """

    max_tokens: float
    max_tool_calls: int
    max_llm_calls: int
    consumed_tokens: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    blocked_llm_calls: int = 0
    blocked_tool_calls: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        """Validate budget parameters and activate counter mutation guard.

        Rejects NaN, Inf, and negative values for limit fields to prevent
        silent bypass of exhaustion checks (IEEE 754: NaN < x is always False).

        After validation, marks initialization complete so the __setattr__
        guard prevents direct writes to counter fields. This means every
        BudgetState instance (whether created directly or via factory) has
        its counters protected after __init__ completes.
        """
        for name, value in (
            ("max_tokens", self.max_tokens),
            ("max_tool_calls", self.max_tool_calls),
            ("max_llm_calls", self.max_llm_calls),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"BudgetState.{name} must be finite, got {value!r}"
                )
            if value < 0:
                raise ValueError(
                    f"BudgetState.{name} must be non-negative, got {value!r}"
                )
        # Activate the counter mutation guard. Done at end of __post_init__
        # so that the dataclass-generated __init__ can still assign initial
        # counter values (consumed_tokens=0.0 etc.) before the guard fires.
        self._mark_init_done()

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent direct writes to counter fields after initialization.

        Counter fields (consumed_tokens, llm_calls, tool_calls,
        blocked_llm_calls, blocked_tool_calls) must be updated via the
        provided async methods to stay lock-protected. Direct assignment
        raises AttributeError after the dataclass __init__ completes.

        The _init_done flag is set at the end of __init__ by Python's
        dataclass machinery.
        """
        _counter_fields = {
            "consumed_tokens",
            "llm_calls",
            "tool_calls",
            "blocked_llm_calls",
            "blocked_tool_calls",
        }
        if name in _counter_fields and hasattr(self, "_init_done"):
            raise AttributeError(
                f"BudgetState.{name} is read-only after init -- "
                "use record_tokens(), try_consume_llm_call(), or "
                "try_consume_tool_call() to mutate counters safely."
            )
        super().__setattr__(name, value)

    def _unsafe_set(self, name: str, value: object) -> None:
        """Internal helper: set a counter field, bypassing the guard.

        Only for use inside async methods that already hold _lock.
        """
        super().__setattr__(name, value)

    def _mark_init_done(self) -> None:
        """Mark initialization complete; counter fields become read-only."""
        super().__setattr__("_init_done", True)

    @property
    def token_exhausted(self) -> bool:
        """True when consumed_tokens >= max_tokens."""
        return self.consumed_tokens >= self.max_tokens

    @property
    def llm_exhausted(self) -> bool:
        """True when llm_calls >= max_llm_calls."""
        return self.llm_calls >= self.max_llm_calls

    @property
    def tool_exhausted(self) -> bool:
        """True when tool_calls >= max_tool_calls."""
        return self.tool_calls >= self.max_tool_calls

    async def record_tokens(self, amount: float) -> None:
        """Atomically add amount to consumed_tokens.

        This is the safe, lock-protected way to update the token counter.

        Parameters
        ----------
        amount:
            Number of tokens to add. Must be finite and non-negative.
            NaN or Inf would silently corrupt token_exhausted checks
            (NaN >= x is always False in IEEE 754).

        Raises
        ------
        ValueError:
            If amount is NaN, Inf, or negative.
        """
        if not math.isfinite(amount):
            raise ValueError(
                f"record_tokens() amount must be finite, got {amount!r}"
            )
        if amount < 0:
            raise ValueError(
                f"record_tokens() amount must be non-negative, got {amount!r}"
            )
        async with self._lock:
            self._unsafe_set("consumed_tokens", self.consumed_tokens + amount)

    async def try_consume_llm_call(self) -> tuple[bool, str]:
        """Atomically check and consume one LLM call slot.

        Returns (True, "") if consumed, (False, reason) if exhausted.
        reason is one of "tokens" or "llm_calls" and is captured inside
        the lock to avoid TOCTOU races in diagnostic logging.

        Note: the slot is reserved before call_next is invoked. If the
        downstream call fails, the slot is NOT returned (the call attempt
        was made). Use llm_calls as "attempted LLM calls" rather than
        "successful LLM calls".
        """
        async with self._lock:
            if self.token_exhausted or self.llm_exhausted:
                # Capture reason inside lock to avoid TOCTOU in log output.
                reason = "tokens" if self.token_exhausted else "llm_calls"
                self._unsafe_set("blocked_llm_calls", self.blocked_llm_calls + 1)
                return False, reason
            self._unsafe_set("llm_calls", self.llm_calls + 1)
            return True, ""

    async def try_consume_tool_call(self) -> bool:
        """Atomically check and consume one tool call slot.

        Returns True if consumed, False if exhausted.
        """
        async with self._lock:
            if self.tool_exhausted:
                self._unsafe_set("blocked_tool_calls", self.blocked_tool_calls + 1)
                return False
            self._unsafe_set("tool_calls", self.tool_calls + 1)
            return True


def _make_budget_state(
    max_tokens: float,
    max_tool_calls: int,
    max_llm_calls: int,
) -> BudgetState:
    """Factory alias: create a BudgetState.

    Equivalent to BudgetState(...) directly. __post_init__ handles validation
    and activates the counter mutation guard automatically.

    Kept for backwards compatibility and for callers that prefer an explicit
    factory call over direct dataclass construction.
    """
    return BudgetState(
        max_tokens=max_tokens,
        max_tool_calls=max_tool_calls,
        max_llm_calls=max_llm_calls,
    )
