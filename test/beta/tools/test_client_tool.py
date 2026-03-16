# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import ExitStack

import pytest

from autogen.beta.context import Context
from autogen.beta.events import ClientToolCall, ToolCall
from autogen.beta.tools.final.client_tool import ClientTool


class _CapturingStream:
    """Minimal stream implementation that records sent events."""

    def __init__(self) -> None:
        self.sent_events: list = []

    async def send(self, event: object, context: object) -> None:
        self.sent_events.append(event)

    def where(self, condition: object) -> "_CapturingStream":
        return self

    def sub_scope(self, func: object, **kwargs: object) -> "_CapturingStream":
        self._captured_func = func
        return self

    def __enter__(self) -> "_CapturingStream":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


@pytest.mark.asyncio
async def test_client_tool_call_returns_client_tool_call() -> None:
    """ClientTool.__call__ must return a ClientToolCall wrapping the original call."""
    schema = {"function": {"name": "my_client_tool", "description": "desc", "parameters": {}}}
    client_tool = ClientTool(schema)

    stream = _CapturingStream()
    context = Context(stream=stream)  # type: ignore[arg-type]

    call = ToolCall(name="my_client_tool", arguments="{}")
    result = await client_tool(call, context)

    assert isinstance(result, ClientToolCall)
    assert result.name == "my_client_tool"
    assert result.parent_id == call.id


@pytest.mark.asyncio
async def test_client_tool_register_execute_sends_to_stream() -> None:
    """The execute closure inside register() must send ClientToolCall to the stream.

    Regression: the original code did `return await execution(...)` without
    `await context.send(result)`, so ToolExecutor.execute_tools() would block
    forever waiting for a ClientToolCall that was never sent to the stream.
    """
    schema = {"function": {"name": "my_tool", "description": "desc", "parameters": {}}}
    client_tool = ClientTool(schema)

    stream = _CapturingStream()
    context = Context(stream=stream)  # type: ignore[arg-type]

    with ExitStack() as stack:
        client_tool.register(stack, context)

    assert hasattr(stream, "_captured_func"), "sub_scope was never called -- register() is broken"

    call = ToolCall(name="my_tool", arguments="{}")
    await stream._captured_func(call, context)

    assert len(stream.sent_events) == 1, (
        f"Expected 1 event sent to stream, got {len(stream.sent_events)}. "
        "ClientTool.register() execute callback must call context.send(result)."
    )
    sent = stream.sent_events[0]
    assert isinstance(sent, ClientToolCall), f"Expected ClientToolCall, got {type(sent)}"
    assert sent.parent_id == call.id
    assert sent.name == call.name


@pytest.mark.asyncio
async def test_client_tool_register_with_middleware() -> None:
    """execute closure must propagate through middleware before sending."""
    schema = {"function": {"name": "mw_tool", "description": "desc", "parameters": {}}}
    client_tool = ClientTool(schema)

    stream = _CapturingStream()
    context = Context(stream=stream)  # type: ignore[arg-type]

    class TagMiddleware:
        async def on_tool_execution(self, call_next: object, event: object, context: object) -> object:
            result = await call_next(event, context)  # type: ignore[misc]
            result._tag = "middleware_ran"  # type: ignore[attr-defined]
            return result

    with ExitStack() as stack:
        client_tool.register(stack, context, middleware=[TagMiddleware()])

    call = ToolCall(name="mw_tool", arguments="{}")
    await stream._captured_func(call, context)

    assert len(stream.sent_events) == 1
    sent = stream.sent_events[0]
    assert isinstance(sent, ClientToolCall)
    assert getattr(sent, "_tag", None) == "middleware_ran"
