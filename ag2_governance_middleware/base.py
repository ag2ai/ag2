"""Base middleware class and onion-chain builder for AG2.1 governance middleware."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from typing import Any


# Type alias for the "call next layer" callable.
# Matches the LLMClient protocol signature.
CallNext = Callable[..., Awaitable[None]]


class BaseMiddleware(ABC):
    """
    Base class for AG2.1 governance middleware.

    Middleware intercepts LLM calls and tool calls by implementing on_llm_call()
    and on_tool_call(). The middleware chain is built as an onion: the first
    element in the middleware list wraps all subsequent ones.

    Each on_* method receives call_next, which invokes the next layer.
    Short-circuiting is achieved by NOT calling call_next and instead emitting
    a response directly via ctx.send().
    """

    @abstractmethod
    async def on_llm_call(
        self,
        call_next: CallNext,
        *messages: Any,
        ctx: Any,
        tools: Iterable[Any],
    ) -> None:
        """
        Intercept an LLM call.

        Parameters
        ----------
        call_next:
            Invoke the next middleware layer (or the real LLMClient).
        messages:
            Conversation history events (BaseEvent instances).
        ctx:
            Execution context. Use ctx.send() to emit short-circuit responses.
        tools:
            Tools available to the model.
        """
        ...

    async def on_tool_call(
        self,
        call_next: Callable[[Any, Any], Awaitable[None]],
        event: Any,
        ctx: Any,
    ) -> None:
        """
        Intercept a single tool call.

        Default implementation is a pure pass-through. Override to add
        policy or budget checks.

        Parameters
        ----------
        call_next:
            Invoke the next handler. Signature: call_next(event, ctx).
        event:
            The ToolCall event being intercepted.
        ctx:
            Execution context.
        """
        await call_next(event, ctx)


def build_middleware_client(
    real_client: Any,
    middleware_stack: list[BaseMiddleware],
) -> CallNext:
    """
    Wrap real_client in an onion chain of middleware.

    Parameters
    ----------
    real_client:
        The actual LLMClient (callable matching LLMClient protocol).
    middleware_stack:
        List of middleware instances. Index 0 = outermost (called first).

    Returns
    -------
    A callable with the same signature as LLMClient that threads calls
    through the middleware chain before reaching real_client.

    Notes
    -----
    An empty middleware_stack is valid and returns a thin wrapper around
    real_client with no interception. This is useful for testing and for
    configurations where middleware is conditionally applied.
    """

    async def innermost(*messages: Any, ctx: Any, tools: Iterable[Any]) -> None:
        await real_client(*messages, ctx=ctx, tools=tools)

    current: CallNext = innermost

    # Build inside-out: reversed() ensures list[0] wraps everything else.
    for mw in reversed(middleware_stack):

        def _make_wrapper(middleware: BaseMiddleware, next_layer: CallNext) -> CallNext:
            async def wrapper(*messages: Any, ctx: Any, tools: Iterable[Any]) -> None:
                await middleware.on_llm_call(next_layer, *messages, ctx=ctx, tools=tools)

            return wrapper

        current = _make_wrapper(mw, current)

    return current


def build_middleware_tool_chain(
    real_handler: Callable[[Any, Any], Awaitable[None]],
    middleware_stack: list[BaseMiddleware],
) -> Callable[[Any, Any], Awaitable[None]]:
    """
    Wrap real_handler in an onion chain of middleware on_tool_call hooks.

    Parameters
    ----------
    real_handler:
        The actual tool executor. Signature: (event, ctx) -> None.
    middleware_stack:
        List of middleware instances. Index 0 = outermost (called first).

    Returns
    -------
    A callable (event, ctx) -> None that threads calls through the
    middleware chain before reaching real_handler.

    Notes
    -----
    An empty middleware_stack is valid and returns real_handler unwrapped.
    """
    current = real_handler

    for mw in reversed(middleware_stack):

        def _make_wrapper(
            middleware: BaseMiddleware,
            next_layer: Callable[[Any, Any], Awaitable[None]],
        ) -> Callable[[Any, Any], Awaitable[None]]:
            async def wrapper(event: Any, ctx: Any) -> None:
                await middleware.on_tool_call(next_layer, event, ctx)

            return wrapper

        current = _make_wrapper(mw, current)

    return current
