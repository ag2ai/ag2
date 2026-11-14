# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import json
import warnings
from pathlib import Path
from weakref import ref

import pytest
from dirty_equals import AnyThing, IsPartialDataclass, IsPartialDict

from ag2 import Agent, Context, KnowledgeConfig, MemoryStream, observer, tool
from ag2.agent import TaskConfig
from ag2.aggregate import AggregateTrigger
from ag2.compact import CompactTrigger, TailWindowCompact
from ag2.events import (
    BaseEvent,
    CompactionStarted,
    ModelMessage,
    ModelRequest,
    ObserverCompleted,
    ToolCallEvent,
    ToolResultEvent,
    ToolResultsEvent,
)
from ag2.exceptions import HumanInputFailedError
from ag2.knowledge import EventLogWriter, KnowledgePlugin, KnowledgeStore, MemoryKnowledgeStore
from ag2.middleware import BaseMiddleware, Middleware, ToolExecution, ToolResultType, approval_required
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.schemas import ToolSchema
from ag2.tools.types import Tool
from test._helpers import ScriptedHuman


async def ask_with_temporary_store(agent: Agent) -> ref[MemoryKnowledgeStore]:
    store = MemoryKnowledgeStore()
    reference = ref(store)
    reply = await agent.ask("use temporary memory", dependencies={KnowledgeStore: store})
    assert reply.body == "done"
    assert await store.exists("/.initialized")
    return reference


@pytest.mark.asyncio
async def test_completed_invocations_release_overridden_stores() -> None:
    agent = Agent("temporary", config=TestConfig("done"), plugins=[KnowledgePlugin(MemoryKnowledgeStore())])
    references = [await ask_with_temporary_store(agent) for _ in range(10)]
    await asyncio.sleep(0)
    gc.collect()

    assert all(reference() is None for reference in references)
    reply = await agent.ask("agent still works")
    assert reply.body == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation_override", [False, True])
async def test_plugin_uses_effective_store_for_tool_bootstrap_and_logging(invocation_override: bool) -> None:
    default = MemoryKnowledgeStore()
    effective = MemoryKnowledgeStore()
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="knowledge", arguments='{"action":"write","path":"/note","content":"hello"}'),
            "done",
        )
    )
    agent = Agent(
        "writer",
        config=config,
        plugins=[
            KnowledgePlugin(
                default,
                aggregate=RememberHistory(),
                aggregate_trigger=AggregateTrigger(on_end=True),
                compact=TailWindowCompact(1),
                compact_trigger=CompactTrigger(max_events=1),
            )
        ],
        dependencies={} if invocation_override else {KnowledgeStore: effective},
    )
    assert await effective.list("/") == []

    reply = await agent.ask("Remember hello", dependencies={KnowledgeStore: effective} if invocation_override else {})

    assert reply.body == "done"
    assert config.calls[0] == IsPartialDataclass(
        dependencies=IsPartialDict({KnowledgeStore: effective}),
        tools=(
            IsPartialDataclass(
                function=IsPartialDataclass(
                    name="knowledge",
                    parameters=IsPartialDict({
                        "properties": {"action": AnyThing, "path": AnyThing, "content": AnyThing},
                    }),
                ),
            ),
        ),
    )
    assert await effective.read("/note") == "hello"
    assert await effective.read("/remembered") == "Remember hello"
    assert await effective.read(f"/log/{reply.context.stream.id}.dropped-1.jsonl") is not None
    assert "writer" in await effective.read("/SKILL.md")
    assert await effective.read(f"/log/{reply.context.stream.id}.jsonl") is not None
    assert await default.list("/") == []


class RememberHistory:
    async def aggregate(self, events: list[BaseEvent], context: Context, store: KnowledgeStore) -> None:
        await store.write("/remembered", "|".join(e.parts[0].content for e in events if isinstance(e, ModelRequest)))


@observer(ObserverCompleted)
async def completion_note(event: ObserverCompleted, ctx: Context) -> None:
    await ctx.send(ModelMessage("observer finished"))


@pytest.mark.asyncio
async def test_aggregation_sees_history_before_compaction_and_log_includes_observer_completion() -> None:
    store = MemoryKnowledgeStore()
    agent = Agent(
        "maintainer",
        config=TestConfig("done"),
        plugins=[
            KnowledgePlugin(
                store,
                aggregate=RememberHistory(),
                aggregate_trigger=AggregateTrigger(on_end=True),
                compact=TailWindowCompact(target=1),
                compact_trigger=CompactTrigger(max_events=1),
            )
        ],
        observers=[completion_note],
    )
    reply = await agent.ask("remember this", stream=MemoryStream(persist_all=True))

    assert await store.read("/remembered") == "remember this"
    history = await reply.context.stream.history.get_events()
    assert not any(isinstance(e, ModelRequest) and e.parts[0].content == "remember this" for e in history)
    log = await EventLogWriter(store).load(reply.context.stream.id)
    assert any(isinstance(e, ObserverCompleted) for e in log)
    assert any(isinstance(e, ModelMessage) and e.content == "observer finished" for e in log)


@pytest.mark.asyncio
async def test_legacy_warning_reports_constructor_callsite_once() -> None:
    store = MemoryKnowledgeStore()
    with pytest.warns(DeprecationWarning) as records:
        agent = Agent("legacy", config=TestConfig("ok", "again"), knowledge=KnowledgeConfig(store))
    [record] = records
    assert Path(record.filename) == Path(__file__)
    assert all(text in str(record.message) for text in ("Agent(knowledge=...)", "KnowledgePlugin", "plugins", "1.4.0"))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        reply = await agent.ask("one")
        assert (await reply.ask("two")).body == "again"
    assert await store.exists("/.initialized")


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation", [False, True])
async def test_mixed_legacy_and_plugin_rejected_before_store_io(invocation: bool) -> None:
    legacy = MemoryKnowledgeStore()
    new = MemoryKnowledgeStore()
    plugin = KnowledgePlugin(new)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="knowledge.*KnowledgePlugin"):
        agent = Agent(
            "mixed",
            config=TestConfig("ok"),
            knowledge=KnowledgeConfig(legacy),
            plugins=[] if invocation else [plugin],
        )
        await agent.ask("hello", plugins=[plugin])
    assert await legacy.list("/") == []
    assert await new.list("/") == []


async def write_during_prompt(ctx: Context) -> str:
    await ctx.dependencies[KnowledgeStore].write("/before-rejection", "unexpected")
    return "prompt"


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation_prompt", [False, True])
async def test_mixed_configuration_rejected_before_dynamic_prompts(invocation_prompt: bool) -> None:
    legacy = MemoryKnowledgeStore()
    new = MemoryKnowledgeStore()
    plugin = KnowledgePlugin(new)
    with pytest.warns(DeprecationWarning):
        agent = Agent("mixed", config=TestConfig("unreachable"), knowledge=KnowledgeConfig(legacy))
    (plugin if invocation_prompt else agent).prompt(write_during_prompt)
    with pytest.raises(ValueError, match="knowledge.*KnowledgePlugin"):
        await agent.ask("hello", plugins=[plugin])
    assert await legacy.list("/") == []
    assert await new.list("/") == []


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::DeprecationWarning")
async def test_agent_without_knowledge_gets_no_store_or_tool() -> None:
    config = TrackingConfig(TestConfig("ok"))
    agent = Agent("bare", config=config)

    await agent.ask("hi")

    [call] = config.calls
    assert call == IsPartialDataclass(tools=())
    assert KnowledgeStore not in call.dependencies


@pytest.mark.asyncio
@pytest.mark.filterwarnings("error::DeprecationWarning")
async def test_knowledge_none_does_not_warn_or_add_store() -> None:
    config = TrackingConfig(TestConfig("ok"))
    agent = Agent("bare", config=config, knowledge=None)

    await agent.ask("hi")

    [call] = config.calls
    assert call == IsPartialDataclass(tools=())
    assert KnowledgeStore not in call.dependencies


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "reply.ask", "reply.run"])
async def test_invocation_plugin_is_scoped_to_its_turn(entry: str) -> None:
    store = MemoryKnowledgeStore()
    config = TrackingConfig(TestConfig("first", "second", "third"))
    agent = Agent("caller", config=config)
    plugin = KnowledgePlugin(store)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if entry.startswith("reply"):
            previous = await agent.ask("start")
            if entry == "reply.ask":
                reply = await previous.ask("with knowledge", plugins=[plugin])
            else:
                async with previous.run("with knowledge", plugins=[plugin]) as run:
                    reply = await run.result()
        elif entry == "run":
            async with agent.run("with knowledge", plugins=[plugin]) as run:
                run.start()
                reply = await run.result()
        else:
            reply = await agent.ask("with knowledge", plugins=[plugin])
    assert config.calls[-1] == IsPartialDataclass(
        dependencies=IsPartialDict({KnowledgeStore: store}),
        tools=(IsPartialDataclass(function=IsPartialDataclass(name="knowledge")),),
    )
    context = reply.context
    assert KnowledgeStore not in context.dependencies
    log_before = await store.read(f"/log/{reply.context.stream.id}.jsonl")
    await reply.ask("without knowledge")
    last_call = config.calls[-1]
    assert last_call == IsPartialDataclass(tools=())
    assert KnowledgeStore not in last_call.dependencies
    assert await store.read(f"/log/{reply.context.stream.id}.jsonl") == log_before
    assert agent.tools == []


class SlowBootstrap:
    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        await store.write("/custom", actor_name)


class ReadInitialized(Tool):
    name = "initialized"

    async def schemas(self, context: Context) -> list[ToolSchema]:
        store = context.dependencies[KnowledgeStore]
        assert await store.read("/custom") == "booter"
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_concurrent_calls_wait_for_custom_bootstrap_before_schema_resolution(override: bool) -> None:
    bootstrap = SlowBootstrap()
    configured = MemoryKnowledgeStore()
    store = MemoryKnowledgeStore() if override else configured
    dependencies = {KnowledgeStore: store}
    agent = Agent(
        "booter",
        config=TestConfig("ok"),
        tools=[ReadInitialized()],
        plugins=[KnowledgePlugin(configured, bootstrap=bootstrap)],
    )
    assert await store.list("/") == []
    first = asyncio.create_task(agent.ask("first", stream=MemoryStream(), dependencies=dependencies))
    await bootstrap.entered.wait()
    second = asyncio.create_task(agent.ask("second", stream=MemoryStream(), dependencies=dependencies))
    await asyncio.sleep(0)
    assert not first.done() and not second.done()
    bootstrap.release.set()
    await asyncio.gather(first, second)
    await agent.ask("third", dependencies=dependencies)
    assert bootstrap.calls == 1


class ApproveKnowledge(BaseMiddleware):
    async def on_tool_execution(
        self, call_next: ToolExecution, event: ToolCallEvent, context: Context
    ) -> ToolResultType:
        return await approval_required()(call_next, event, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["write", "delete"])
@pytest.mark.parametrize("answer", ["yes", "no"])
async def test_tool_approval_controls_action_and_maintenance_still_runs(action: str, answer: str) -> None:
    store = MemoryKnowledgeStore()
    await store.write("/target", "old")
    agent = Agent(
        "approved",
        config=TestConfig(
            ToolCallEvent(
                name="knowledge", arguments=json.dumps({"action": action, "path": "/target", "content": "new"})
            ),
            "done",
        ),
        plugins=[KnowledgePlugin(store, aggregate=RememberHistory(), aggregate_trigger=AggregateTrigger(on_end=True))],
        middleware=[Middleware(ApproveKnowledge)],
        hitl_hook=ScriptedHuman(answer),
    )
    reply = await agent.ask("maintain memory")
    expected = ("new" if action == "write" else None) if answer == "yes" else "old"
    assert await store.read("/target") == expected
    assert await store.read("/remembered") == "maintain memory"
    assert await store.exists("/.initialized")
    assert await store.read(f"/log/{reply.context.stream.id}.jsonl") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("model failed"), asyncio.CancelledError()])
async def test_failure_and_cancellation_aggregate_without_compaction_or_final_log(failure: BaseException) -> None:
    store = MemoryKnowledgeStore()
    stream = MemoryStream(persist_all=True)
    agent = Agent(
        "failing",
        config=TestConfig("initial", failure),
    )
    previous = await agent.ask("initial", stream=stream)
    plugin = KnowledgePlugin(
        store,
        aggregate=RememberHistory(),
        aggregate_trigger=AggregateTrigger(on_end=True),
        compact=TailWindowCompact(1),
        compact_trigger=CompactTrigger(max_events=1),
    )
    with pytest.raises(type(failure)):
        await previous.ask("remember failed attempt", plugins=[plugin], observers=[completion_note])
    assert await store.read("/remembered") == "initial|remember failed attempt"
    assert await store.read(f"/log/{stream.id}.jsonl") is None
    assert any(isinstance(e, ObserverCompleted) for e in await stream.history.get_events())
    assert not any(isinstance(e, CompactionStarted) for e in await stream.history.get_events())


@pytest.mark.asyncio
async def test_invocation_tool_overrides_plugin_knowledge_once() -> None:
    store = MemoryKnowledgeStore()
    override = tool(replace_knowledge, name="knowledge")
    call = ToolCallEvent(name="knowledge", arguments="{}")
    config = TrackingConfig(TestConfig(call, "done"))
    reply = await Agent("override", config=config, plugins=[KnowledgePlugin(store)]).ask(
        "use override", tools=[override]
    )
    [results] = [e for e in config.calls[-1].messages if isinstance(e, ToolResultsEvent)]
    assert reply.body == "done"
    assert results == ToolResultsEvent([ToolResultEvent.from_call(call, "override")])
    assert config.calls[0] == IsPartialDataclass(
        tools=(IsPartialDataclass(function=IsPartialDataclass(name="knowledge")),),
    )


def replace_knowledge() -> str:
    return "override"


@pytest.mark.asyncio
async def test_parent_plugin_does_not_equip_explicit_worker() -> None:
    worker_config = TrackingConfig(TestConfig("worker done"))
    worker = Agent("worker", config=worker_config)
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"work"}'), "done"),
        plugins=[KnowledgePlugin(MemoryKnowledgeStore())],
        tools=[worker.as_tool(description="Work")],
    )
    await root.ask("delegate")
    assert worker_config.calls[0] == IsPartialDataclass(tools=())


@pytest.mark.asyncio
async def test_failed_run_still_writes_event_log() -> None:
    store = MemoryKnowledgeStore()
    stream = MemoryStream()
    agent = Agent("failing", config=TestConfig(RuntimeError("failed")), plugins=[KnowledgePlugin(store)])

    with pytest.raises(RuntimeError, match="failed"):
        async with agent.run("failed attempt", stream=stream) as run:
            await run.result()

    assert await store.read(f"/log/{stream.id}.jsonl") is not None


@pytest.mark.asyncio
@pytest.mark.filterwarnings(r"ignore:Agent\(knowledge=:DeprecationWarning")
async def test_legacy_knowledge_failed_run_still_writes_event_log() -> None:
    store = MemoryKnowledgeStore()
    stream = MemoryStream()
    agent = Agent("failing", config=TestConfig(RuntimeError("failed")), knowledge=KnowledgeConfig(store))

    with pytest.raises(RuntimeError, match="failed"):
        async with agent.run("failed attempt", stream=stream) as run:
            await run.result()

    assert await store.read(f"/log/{stream.id}.jsonl") is not None


@pytest.mark.asyncio
@pytest.mark.filterwarnings("ignore:TaskConfig, Agent:DeprecationWarning")
async def test_legacy_precedence_and_hidden_tool_are_preserved() -> None:
    configured = MemoryKnowledgeStore()
    constructor_override = MemoryKnowledgeStore()
    invocation_override = MemoryKnowledgeStore()
    worker_config = TrackingConfig(TestConfig("worker done"))
    config = TrackingConfig(
        TestConfig(
            ToolCallEvent(name="knowledge", arguments='{"action":"write","path":"/note","content":"legacy"}'),
            ToolCallEvent(name="run_subtask", arguments='{"task":"work"}'),
            "done",
        )
    )
    with pytest.warns(DeprecationWarning, match=r"Agent\(knowledge="):
        agent = Agent(
            "legacy",
            config=config,
            knowledge=KnowledgeConfig(configured),
            dependencies={KnowledgeStore: constructor_override},
            tasks=TaskConfig(config=worker_config),
        )
    assert agent.dependencies == IsPartialDict({KnowledgeStore: configured})
    await agent.ask("use legacy", dependencies={KnowledgeStore: invocation_override})
    assert config.calls[0] == IsPartialDataclass(dependencies=IsPartialDict({KnowledgeStore: invocation_override}))
    assert await configured.read("/note") == "legacy"
    assert await constructor_override.list("/") == []
    assert await invocation_override.list("/") == []
    assert worker_config.calls[0] == IsPartialDataclass(tools=())


async def broken_approval(request: object) -> str:
    raise OSError("approval channel unavailable")


async def cancelled_approval(request: object) -> str:
    raise asyncio.CancelledError


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_approval_channel_failure_preserves_continuation_cleanup(cancel: bool) -> None:
    store = MemoryKnowledgeStore()
    stream = MemoryStream()
    agent = Agent(
        "approval",
        config=TestConfig(
            "initial",
            ToolCallEvent(
                name="knowledge",
                arguments='{"action":"write","path":"/target","content":"new"}',
            ),
        ),
    )
    previous = await agent.ask("initial", stream=stream)
    plugin = KnowledgePlugin(store, aggregate=RememberHistory(), aggregate_trigger=AggregateTrigger(on_end=True))
    with pytest.raises(asyncio.CancelledError if cancel else HumanInputFailedError):
        await previous.ask(
            "remember attempt",
            plugins=[plugin],
            middleware=[Middleware(ApproveKnowledge)],
            hitl_hook=cancelled_approval if cancel else broken_approval,
        )
    assert await store.read("/target") is None
    assert await store.read("/remembered") == "initial|remember attempt"
    assert await store.read(f"/log/{stream.id}.jsonl") is None


class BrokenBootstrap:
    async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
        raise OSError("bootstrap failed")


@pytest.mark.asyncio
async def test_bootstrap_failure_propagates_before_model_execution() -> None:
    store = MemoryKnowledgeStore()
    config = TrackingConfig(TestConfig("unreachable"))
    agent = Agent("broken", config=config, plugins=[KnowledgePlugin(store, bootstrap=BrokenBootstrap())])
    with pytest.raises(OSError, match="bootstrap failed"):
        await agent.ask("start")
    assert config.calls == []
    assert await store.list("/log") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invocation", [False, True])
async def test_second_plugin_rejected_before_store_io(invocation: bool) -> None:
    first = MemoryKnowledgeStore()
    second = MemoryKnowledgeStore()
    with pytest.raises(ValueError, match="Only one KnowledgePlugin"):
        agent = Agent(
            "doubled",
            config=TestConfig("unreachable"),
            plugins=[KnowledgePlugin(first)] if invocation else [KnowledgePlugin(first), KnowledgePlugin(second)],
        )
        await agent.ask("hello", plugins=[KnowledgePlugin(second)])
    assert await first.list("/") == []
    assert await second.list("/") == []


@pytest.mark.parametrize(
    "options",
    [
        {"compact": TailWindowCompact(1)},
        {"aggregate_trigger": AggregateTrigger(on_end=True)},
    ],
)
def test_strategy_without_its_counterpart_is_rejected(options: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="requires"):
        KnowledgePlugin(MemoryKnowledgeStore(), **options)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_worker_plugin_store_wins_over_inherited_parent_store() -> None:
    parent_store = MemoryKnowledgeStore()
    worker_store = MemoryKnowledgeStore()
    worker = Agent(
        "worker",
        config=TestConfig(
            ToolCallEvent(name="knowledge", arguments='{"action":"write","path":"/note","content":"worker"}'),
            "worker done",
        ),
        plugins=[KnowledgePlugin(worker_store)],
    )
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"work"}'), "done"),
        plugins=[KnowledgePlugin(parent_store)],
        tools=[worker.as_tool(description="Work")],
    )

    await parent.ask("delegate")

    assert await worker_store.read("/note") == "worker"
    assert "worker" in await worker_store.read("/SKILL.md")
    assert await parent_store.read("/note") is None


@pytest.mark.asyncio
async def test_request_store_overrides_worker_plugin_store() -> None:
    tenant_store = MemoryKnowledgeStore()
    shared_store = MemoryKnowledgeStore()
    worker = Agent(
        "worker",
        config=TestConfig(
            ToolCallEvent(name="knowledge", arguments='{"action":"write","path":"/note","content":"tenant"}'),
            "worker done",
        ),
        plugins=[KnowledgePlugin(shared_store)],
    )
    parent = Agent(
        "parent",
        config=TestConfig(ToolCallEvent(name="task_worker", arguments='{"objective":"work"}'), "done"),
        plugins=[KnowledgePlugin(MemoryKnowledgeStore())],
        tools=[worker.as_tool(description="Work")],
    )

    await parent.ask("delegate", dependencies={KnowledgeStore: tenant_store})

    assert await tenant_store.read("/note") == "tenant"
    assert await shared_store.read("/note") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [False, True])
async def test_failed_bootstrap_is_retried(override: bool) -> None:
    class FlakyBootstrap:
        def __init__(self) -> None:
            self.attempts = 0

        async def bootstrap(self, store: KnowledgeStore, actor_name: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("bootstrap temporarily unavailable")
            await store.write("/ready", actor_name)

    bootstrap = FlakyBootstrap()
    store = MemoryKnowledgeStore()
    plugin = KnowledgePlugin(
        MemoryKnowledgeStore() if override else store,
        bootstrap=bootstrap,
        write_event_log=False,
    )
    agent = Agent("initializer", config=TestConfig("ok"), plugins=[plugin])
    dependencies = {KnowledgeStore: store} if override else {}

    with pytest.raises(OSError, match="temporarily unavailable"):
        await agent.ask("initialize", dependencies=dependencies)

    await agent.ask("retry", dependencies=dependencies)
    assert await store.read("/ready") == "initializer"

    await agent.ask("already initialized", dependencies=dependencies)
    assert bootstrap.attempts == 2
