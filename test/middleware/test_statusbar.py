# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any

import pytest

from ag2 import Agent, Context, DataInput, Depends, Inject, MemoryStream, TextInput, Variable
from ag2.events import BaseEvent, Input, ModelRequest, ModelResponse, ToolCallEvent
from ag2.middleware import (
    BaseMiddleware,
    LLMCall,
    MiddlewareDescription,
    RetryMiddleware,
    statusbar_middleware,
)
from ag2.plugin import Plugin
from ag2.testing import TestConfig, TrackingConfig


@statusbar_middleware
def current_status(context: Context) -> TextInput:
    return TextInput(f"step:{context.variables.get('step', 0)}")


def advance(context: Context) -> str:
    context.variables["step"] = context.variables.get("step", 0) + 1
    return "advanced"


def request_texts(events: Sequence[BaseEvent]) -> list[str]:
    return [
        part.content
        for event in events
        if isinstance(event, ModelRequest)
        for part in event.parts
        if isinstance(part, TextInput)
    ]


@pytest.mark.asyncio
async def test_dynamic_status_refreshes_after_tools_without_entering_history() -> None:
    config = TrackingConfig(TestConfig(ToolCallEvent("advance"), "done", "continued"))
    agent = Agent("status", config=config, middleware=[current_status], tools=[advance], prompt="static")

    reply = await agent.ask("start")
    await reply.ask("continue")

    assert [request_texts(call.messages) for call in config.calls] == [
        ["start", "step:0"],
        ["start", "step:1"],
        ["start", "continue", "step:1"],
    ]
    assert request_texts(await reply.context.stream.history.get_events()) == ["start", "continue"]
    assert [call.prompt for call in config.calls] == [("static",)] * 3


def source(context: Context, label: Annotated[str, Inject()]) -> str:
    return f"{label}:{context.variables.get('step', 0)}"


def overridden_source(label: Annotated[str, Inject()]) -> str:
    return f"override:{label}"


@statusbar_middleware(persist=False)
async def injected_status(
    context: Context,
    label: Annotated[str, Inject()],
    step: Annotated[int, Variable(default=0)],
    value: Annotated[str, Depends(source)],
) -> Sequence[Input]:
    assert context.dependencies["label"] == label
    return [TextInput(f"{label}:{step}"), DataInput({"value": value})]


@pytest.mark.asyncio
async def test_async_status_resolves_injected_variables_and_composed_dependencies_afresh() -> None:
    config = TrackingConfig(TestConfig(ToolCallEvent("advance"), "done"))
    agent = Agent("status", config=config, middleware=[injected_status], tools=[advance])

    await agent.ask("start", dependencies={"label": "job"})

    assert [call.messages[-1] for call in config.calls] == [
        ModelRequest([TextInput("job:0"), DataInput({"value": "job:0"})]),
        ModelRequest([TextInput("job:1"), DataInput({"value": "job:1"})]),
    ]


@pytest.mark.asyncio
async def test_status_uses_agent_dependency_overrides() -> None:
    config = TrackingConfig(TestConfig("done"))
    agent = Agent("status", config=config, middleware=[injected_status])
    agent.dependency_provider.override(source, overridden_source)

    await agent.ask("start", dependencies={"label": "job"})

    assert config.calls[0].messages[-1] == ModelRequest([TextInput("job:0"), DataInput({"value": "override:job"})])


@statusbar_middleware
def supplied_status(value: Annotated[Any, Inject()]) -> Any:
    return value


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["text", {"text": "value"}, [TextInput("valid"), "invalid"], b"bytes", 12])
async def test_invalid_status_fails_before_model_or_history_insertion(value: Any) -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[supplied_status])
    stream = MemoryStream()

    with pytest.raises(TypeError, match="Input, Sequence"):
        await agent.ask("start", dependencies={"value": value}, stream=stream)

    assert config.calls == []
    assert request_texts(await stream.history.get_events()) == ["start"]


@pytest.mark.asyncio
async def test_generator_status_is_rejected_without_consuming_it() -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[supplied_status])
    value = (TextInput(str(i)) for i in range(2))

    with pytest.raises(TypeError, match="Input, Sequence"):
        await agent.ask("start", dependencies={"value": value})

    assert next(value) == TextInput("0")
    assert config.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, [], ()])
async def test_empty_status_leaves_model_input_unchanged(value: Any) -> None:
    config = TrackingConfig(TestConfig("done"))
    agent = Agent("status", config=config, middleware=[supplied_status])

    await agent.ask("start", dependencies={"value": value})

    assert config.calls[0].messages == (ModelRequest([TextInput("start")]),)


@statusbar_middleware
def first_status() -> TextInput:
    return TextInput("first")


@statusbar_middleware
def second_status() -> TextInput:
    return TextInput("second")


@pytest.mark.asyncio
@pytest.mark.parametrize("via_plugin", [False, True])
async def test_invocation_registration_and_plugin_scope(via_plugin: bool) -> None:
    config = TrackingConfig(TestConfig("one", "two", "three"))
    agent = Agent("status", config=config, middleware=[first_status])
    options = {"plugins": [Plugin(middleware=[second_status])]} if via_plugin else {"middleware": [second_status]}

    reply = await agent.ask("start", **options)
    reply = await reply.ask("continue")
    await reply.ask("again", **options)

    assert [request_texts(call.messages) for call in config.calls] == [
        ["start", "first", "second"],
        ["start", "continue", "first"],
        ["start", "continue", "again", "first", "second"],
    ]


@pytest.mark.asyncio
async def test_status_is_available_during_observable_run() -> None:
    config = TrackingConfig(TestConfig("done"))
    agent = Agent("status", config=config, middleware=[first_status])

    async with agent.run("start") as run:
        seen = []
        with run.stream.join() as events:
            run.start()
            async for event in events:
                seen.append(event)
                if isinstance(event, ModelResponse):
                    break
        reply = await run.result()

    assert reply.body == "done"
    assert not any(isinstance(event, ModelRequest) and TextInput("first") in event.parts for event in seen)
    assert request_texts(config.calls[0].messages) == ["start", "first"]


@statusbar_middleware
def counting_status(context: Context) -> TextInput:
    count = context.variables.get("status_count", 0) + 1
    context.variables["status_count"] = count
    return TextInput(f"count:{count}")


@pytest.mark.asyncio
async def test_status_outside_retry_reuses_snapshot_but_refreshes_after_tool_round() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("retry"), ToolCallEvent("advance"), "done"))
    agent = Agent(
        "status",
        config=config,
        tools=[advance],
        middleware=[counting_status, RetryMiddleware(max_retries=1, retry_on=(TimeoutError,))],
    )

    await agent.ask("start")

    assert [request_texts(call.messages) for call in config.calls] == [
        ["start", "count:1"],
        ["start", "count:1"],
        ["start", "count:2"],
    ]


@pytest.mark.asyncio
async def test_status_inside_retry_follows_normal_wrapper_order() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("retry"), "done"))
    agent = Agent(
        "status",
        config=config,
        middleware=[RetryMiddleware(max_retries=1, retry_on=(TimeoutError,)), counting_status],
    )

    await agent.ask("start")

    assert [request_texts(call.messages) for call in config.calls] == [["start", "count:1"], ["start", "count:2"]]


@pytest.mark.asyncio
async def test_concurrent_conversations_on_one_agent_have_independent_status_state() -> None:
    config = TrackingConfig(TestConfig("done", "done"))
    agent = Agent("status", config=config, middleware=[counting_status])

    await asyncio.gather(agent.ask("one"), agent.ask("two"))

    assert sorted(request_texts(call.messages) for call in config.calls) == [
        ["one", "count:1"],
        ["two", "count:1"],
    ]


@dataclass
class Resource:
    events: list[str] = field(default_factory=list)
    opened: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


async def open_resource(resource: Annotated[Resource, Inject()]) -> AsyncGenerator[Resource]:
    resource.opened = True
    resource.events.append("enter")
    try:
        yield resource
    finally:
        resource.events.append("exit")
        resource.opened = False


@statusbar_middleware
async def resource_status(resource: Annotated[Resource, Depends(open_resource)]) -> TextInput:
    resource.events.append("callback")
    return TextInput("resource ready")


class ResourceModelMiddleware(BaseMiddleware):
    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        resource = context.dependencies["resource"]
        assert resource.opened
        resource.events.append("model")
        if context.variables.get("wait"):
            resource.started.set()
            await resource.release.wait()
        return await call_next(events, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_model", [False, True])
async def test_generator_dependency_stays_open_through_model_attempts(fail_model: bool) -> None:
    config = TrackingConfig(TestConfig(ValueError("provider") if fail_model else "done"))
    agent = Agent("status", config=config, middleware=[resource_status, ResourceModelMiddleware])
    resource = Resource()

    if fail_model:
        with pytest.raises(ValueError, match="provider"):
            await agent.ask("start", dependencies={"resource": resource})
    else:
        await agent.ask("start", dependencies={"resource": resource})

    assert resource.events == ["enter", "callback", "model", "exit"]
    assert not resource.opened


@statusbar_middleware
async def broken_resource_status(resource: Annotated[Resource, Depends(open_resource)]) -> TextInput:
    raise ValueError("callback")


@statusbar_middleware
async def invalid_resource_status(resource: Annotated[Resource, Depends(open_resource)]) -> Any:
    return "invalid"


async def broken_dependency(resource: Annotated[Resource, Depends(open_resource)]) -> None:
    raise ValueError("dependency")


@statusbar_middleware
async def broken_dependency_status(value: Annotated[None, Depends(broken_dependency)]) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("middleware", "error"),
    [
        (broken_resource_status, ValueError),
        (invalid_resource_status, TypeError),
        (broken_dependency_status, ValueError),
    ],
)
async def test_resource_cleanup_on_callback_validation_or_dependency_failure(middleware, error) -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[middleware])
    resource = Resource()

    with pytest.raises(error):
        await agent.ask("start", dependencies={"resource": resource})

    assert resource.events == ["enter", "exit"]
    assert not resource.opened
    assert config.calls == []


@pytest.mark.asyncio
async def test_cancellation_closes_status_dependency_and_propagates() -> None:
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("status", config=config, middleware=[resource_status, ResourceModelMiddleware])
    resource = Resource()
    task = asyncio.create_task(agent.ask("start", dependencies={"resource": resource}, variables={"wait": True}))
    await resource.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert resource.events == ["enter", "callback", "model", "exit"]
    assert not resource.opened
    assert config.calls == []


def test_status_description_has_callback_identity_without_dependency_values() -> None:
    agent = Agent("status", middleware=[injected_status], dependencies={"label": "secret"})

    assert agent.middleware[0].description == MiddlewareDescription(
        kind="statusbar_middleware",
        config={"persist": False, "callback": f"{__name__}.injected_status"},
    )


@pytest.mark.asyncio
async def test_dependency_stays_open_through_inner_retries_and_closes_once() -> None:
    config = TrackingConfig(TestConfig(TimeoutError("retry"), "done"))
    agent = Agent(
        "status",
        config=config,
        middleware=[resource_status, RetryMiddleware(max_retries=1, retry_on=(TimeoutError,)), ResourceModelMiddleware],
    )
    resource = Resource()

    await agent.ask("start", dependencies={"resource": resource})

    assert resource.events == ["enter", "callback", "model", "model", "exit"]
    assert not resource.opened


def enqueue_update(context: Context) -> str:
    context.enqueue("background update")
    return "queued"


@statusbar_middleware
async def last_input_status(context: Context) -> TextInput:
    events = await context.stream.history.get_events()
    requests = [event for event in events if isinstance(event, ModelRequest)]
    return TextInput(f"latest:{requests[-1].parts[0].content}")


@pytest.mark.asyncio
async def test_status_observes_pending_input_drained_before_next_model_call() -> None:
    config = TrackingConfig(TestConfig(ToolCallEvent("enqueue_update"), "done"))
    agent = Agent("status", config=config, middleware=[last_input_status], tools=[enqueue_update])

    await agent.ask("start")

    assert [request_texts(call.messages) for call in config.calls] == [
        ["start", "latest:start"],
        ["start", "background update", "latest:background update"],
    ]


class MutateStatusMiddleware(BaseMiddleware):
    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        status = events[-1]
        assert isinstance(status, ModelRequest)
        [part] = status.parts
        assert isinstance(part, DataInput)
        part.data["steps"].append("changed by downstream")
        return await call_next(events, context)


@pytest.mark.asyncio
async def test_downstream_request_changes_do_not_mutate_callback_source_payload() -> None:
    config = TrackingConfig(TestConfig("done"))
    value = DataInput({"steps": ["queued"]})
    agent = Agent("status", config=config, middleware=[supplied_status, MutateStatusMiddleware])

    reply = await agent.ask("start", dependencies={"value": value})

    assert value == DataInput({"steps": ["queued"]})
    assert config.calls[0].messages[-1] == ModelRequest([DataInput({"steps": ["queued", "changed by downstream"]})])
    assert request_texts(await reply.context.stream.history.get_events()) == ["start"]


class CallableStatus:
    def __call__(self) -> TextInput:
        return TextInput("instance")

    def __repr__(self) -> str:
        return "secret"


@pytest.mark.asyncio
async def test_callable_instance_uses_type_identity_in_description() -> None:
    middleware = statusbar_middleware(CallableStatus())
    config = TrackingConfig(TestConfig("done"))
    agent = Agent("status", config=config, middleware=[middleware])

    await agent.ask("start")

    assert agent.middleware[0].description == MiddlewareDescription(
        kind="statusbar_middleware",
        config={"persist": False, "callback": f"{__name__}.CallableStatus"},
    )
    assert request_texts(config.calls[0].messages) == ["start", "instance"]
