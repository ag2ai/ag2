# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import ag2.middleware as middleware_pkg
from ag2 import Context
from ag2.events import ToolCallEvent
from ag2.middleware import (
    ApprovalRequired,
    ConditionalMiddleware,
    DescribableMiddleware,
    HistoryLimiter,
    LoggingMiddleware,
    Middleware,
    MiddlewareDescription,
    RetryMiddleware,
    TokenLimiter,
    ToolExecution,
    ToolResultType,
    approval_required,
)


def described(middleware: object) -> MiddlewareDescription:
    """Describe ``middleware`` through a public surface.

    ``ConditionalMiddleware`` reports whatever it wraps, so this exercises how
    the library describes middleware that has not opted in, without the test
    reaching for internals of its own.
    """

    return ConditionalMiddleware(middleware, ToolCallEvent).describe().inner[0]  # type: ignore[arg-type]


async def undescribed_guard(
    call_next: ToolExecution,
    event: ToolCallEvent,
    context: Context,
) -> ToolResultType:
    """A user-written closure-style hook that has not opted in."""
    return await call_next(event, context)


class UndescribedGuard:
    """A user-written class-based hook that has not opted in."""

    def __init__(self, limit: int) -> None:
        self._limit = limit


class RaisingGuard:
    """Middleware whose describe() is broken."""

    def describe(self) -> MiddlewareDescription:
        raise RuntimeError("kaboom")


class WrongTypeGuard:
    """Middleware whose describe() returns something that is not a description."""

    def describe(self) -> MiddlewareDescription:
        return {"kind": "WrongTypeGuard"}  # type: ignore[return-value]


class TestBuiltinsDescribeThemselves:
    def test_token_limiter(self) -> None:
        assert TokenLimiter(max_tokens=100).describe() == MiddlewareDescription(
            kind="TokenLimiter",
            config={"max_tokens": 100, "chars_per_token": 4},
        )

    def test_history_limiter(self) -> None:
        assert HistoryLimiter(max_events=5).describe() == MiddlewareDescription(
            kind="HistoryLimiter",
            config={"max_events": 5},
        )

    def test_retry_middleware_names_exception_types(self) -> None:
        description = RetryMiddleware(max_retries=2, retry_on=(ValueError,)).describe()

        assert description == MiddlewareDescription(
            kind="RetryMiddleware",
            config={"max_retries": 2, "retry_on": ("ValueError",)},
        )

    def test_logging_middleware_reports_logger_name_not_object(self) -> None:
        assert LoggingMiddleware().describe().config == {"logger": "ag2"}

    def test_approval_required(self) -> None:
        description = approval_required(timeout=5, allow_always=False).describe()

        assert description.kind == "ApprovalRequired"
        assert description.complete is True
        assert description.config["timeout"] == 5
        assert description.config["allow_always"] is False


class TestUndescribedMiddleware:
    def test_reports_incomplete_rather_than_guessing(self) -> None:
        assert described(undescribed_guard) == MiddlewareDescription(
            kind="undescribed_guard",
            config={},
            complete=False,
        )

    def test_never_reads_closure_cells(self) -> None:
        def make_guard(limit: int):  # type: ignore[no-untyped-def]
            async def guard(call_next, event, context):  # type: ignore[no-untyped-def]
                return await call_next(event, context)

            return guard

        description = described(make_guard(limit=7))

        assert description.config == {}
        assert description.complete is False

    def test_class_based_middleware_without_describe(self) -> None:
        assert described(UndescribedGuard(limit=3)) == MiddlewareDescription(
            kind="UndescribedGuard",
            config={},
            complete=False,
        )


class TestWrappers:
    def test_middleware_wrapper_reports_wrapped_class_and_option_names(self) -> None:
        assert Middleware(LoggingMiddleware, level=10).describe() == MiddlewareDescription(
            kind="LoggingMiddleware",
            config={"options": ("level",)},
            complete=False,
        )

    def test_middleware_wrapper_never_reports_option_values(self) -> None:
        # The wrapper cannot know whether a caller-supplied option is a secret.
        description = Middleware(LoggingMiddleware, api_key="sk-SECRET-123").describe()

        assert "sk-SECRET-123" not in repr(description)
        assert description.config == {"options": ("api_key",)}
        assert description.complete is False

    def test_conditional_middleware_reports_inner_separately_from_config(self) -> None:
        description = ConditionalMiddleware(TokenLimiter(max_tokens=10), ToolCallEvent).describe()

        assert description.kind == "ConditionalMiddleware"
        assert description.config == {"condition": "TypeCondition"}
        assert description.inner == (
            MiddlewareDescription(kind="TokenLimiter", config={"max_tokens": 10, "chars_per_token": 4}),
        )

    def test_conditional_middleware_propagates_incompleteness(self) -> None:
        assert ConditionalMiddleware(undescribed_guard, ToolCallEvent).describe().complete is False


class TestBrokenDescribe:
    def test_raising_describe_does_not_take_down_the_caller(self) -> None:
        assert described(RaisingGuard()) == MiddlewareDescription(
            kind="RaisingGuard",
            config={},
            complete=False,
        )

    def test_describe_returning_the_wrong_type_degrades_to_incomplete(self) -> None:
        assert described(WrongTypeGuard()) == MiddlewareDescription(
            kind="WrongTypeGuard",
            config={},
            complete=False,
        )


class TestCompletenessPropagation:
    def test_any_incomplete_inner_makes_the_composite_incomplete(self) -> None:
        composite = MiddlewareDescription(
            kind="Composite",
            inner=(
                MiddlewareDescription(kind="A"),
                MiddlewareDescription(kind="B", complete=False),
            ),
        )

        assert composite.complete is False

    def test_all_complete_inners_leave_the_composite_complete(self) -> None:
        composite = MiddlewareDescription(
            kind="Composite",
            inner=(MiddlewareDescription(kind="A"), MiddlewareDescription(kind="B")),
        )

        assert composite.complete is True

    def test_explicit_incompleteness_is_never_overridden(self) -> None:
        composite = MiddlewareDescription(
            kind="Composite",
            complete=False,
            inner=(MiddlewareDescription(kind="A"),),
        )

        assert composite.complete is False


class TestHashing:
    def test_description_is_deliberately_unhashable(self) -> None:
        # config holds arbitrary values, so equality is supported but hashing is not.
        with pytest.raises(TypeError, match="unhashable type: 'MiddlewareDescription'"):
            hash(TokenLimiter(max_tokens=100).describe())


class TestProtocol:
    def test_builtin_satisfies_protocol(self) -> None:
        assert isinstance(TokenLimiter(max_tokens=1), DescribableMiddleware)

    def test_plain_callable_does_not(self) -> None:
        assert not isinstance(undescribed_guard, DescribableMiddleware)


class TestComparison:
    def test_identical_configuration_compares_equal(self) -> None:
        assert TokenLimiter(max_tokens=50).describe() == TokenLimiter(max_tokens=50).describe()

    def test_differing_configuration_compares_unequal(self) -> None:
        assert TokenLimiter(max_tokens=50).describe() != TokenLimiter(max_tokens=99).describe()


class TestApprovalRequiredRemainsUsable:
    def test_factory_returns_the_class(self) -> None:
        assert isinstance(approval_required(), ApprovalRequired)

    def test_is_still_callable_as_tool_middleware(self) -> None:
        # ToolMiddleware is a Callable alias, so an instance must satisfy it.
        assert callable(approval_required())


class TestPublicSurface:
    def test_no_free_describe_function_is_exported(self) -> None:
        # Middleware reports itself; there is no public module-level describe().
        assert not hasattr(middleware_pkg, "describe_middleware")
        assert "describe_middleware" not in middleware_pkg.__all__

    def test_description_types_are_exported(self) -> None:
        assert "MiddlewareDescription" in middleware_pkg.__all__
        assert "DescribableMiddleware" in middleware_pkg.__all__
