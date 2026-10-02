# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytest
from dirty_equals import IsPartialDict

from ag2 import Agent, AgentRun, Context, MemoryStream, observer, tool
from ag2.config import LLMClient
from ag2.events import BaseEvent, HumanMessage, ModelRequest, ModelResponse, ToolCallEvent, ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.middleware import BaseMiddleware, Middleware
from ag2.middleware.base import LLMCall
from ag2.plugin import Plugin
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema
from ag2.tools.skills import LocalRuntime, SkillPlugin


@dataclass
class RecordedCall:
    prompt: list[str]
    names: list[str]
    schemas: list[ToolSchema]
    dependencies: dict[Any, Any]
    variables: dict[str, Any]


class RecordingClient:
    def __init__(
        self,
        client: LLMClient,
        calls: list[RecordedCall],
        started: asyncio.Event,
        gate: asyncio.Event | None = None,
        rendezvous: bool = False,
    ) -> None:
        self.client = client
        self.calls = calls
        self.started = started
        self.gate = gate
        self.rendezvous = rendezvous

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: Context,
        *,
        tools: Sequence[ToolSchema],
        **kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            RecordedCall(
                list(context.prompt),
                [s.function.name for s in tools if isinstance(s, FunctionToolSchema)],
                list(tools),
                dict(context.dependencies),
                dict(context.variables),
            )
        )
        self.started.set()
        if self.gate is not None:
            if self.rendezvous and len(self.calls) == 2:
                self.gate.set()
            await self.gate.wait()
        return await self.client(messages, context=context, tools=tools, **kwargs)


class RecordingConfig(TestConfig):
    def __init__(self, *events: Any, gate: asyncio.Event | None = None, rendezvous: bool = False) -> None:
        super().__init__(*events)
        self.tracking = TrackingConfig(TestConfig(*events))
        self.calls: list[RecordedCall] = []
        self.started = asyncio.Event()
        self.rendezvous = rendezvous
        self.gate = gate

    def create(self) -> RecordingClient:
        self.tracking.config = TestConfig(*self.events)
        return RecordingClient(self.tracking.create(), self.calls, self.started, self.gate, self.rendezvous)


def tool_text(config: RecordingConfig) -> str:
    event = config.tracking.mock.call_args.args[0]
    assert isinstance(event, ToolResultsEvent)
    [result] = event.results
    [part] = result.result.parts
    return part.content


def dynamic_prompt(ctx: Context) -> str:
    return f"dynamic:{ctx.variables['label']}:{ctx.dependencies['source']}"


def read_defaults(ctx: Context) -> str:
    ctx.variables["tool_result"] = "kept"
    return f"{ctx.variables['label']}:{ctx.dependencies['source']}"


async def request_input(ctx: Context) -> str:
    return await ctx.input("Choose", timeout=1)


def plugin_answer() -> HumanMessage:
    return HumanMessage("plugin")


def agent_answer() -> HumanMessage:
    return HumanMessage("agent")


def call_answer() -> HumanMessage:
    return HumanMessage("call")


class SuffixMiddleware(BaseMiddleware):
    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        original = context.prompt
        context.prompt = [*original, "middleware"]
        try:
            return await call_next(events, context)
        finally:
            context.prompt = original


@dataclass
class AppendPolicy:
    name: str

    async def apply(
        self, prompts: list[str], events: list[BaseEvent], context: Context
    ) -> tuple[list[str], list[BaseEvent]]:
        return [*prompts, self.name], events


@dataclass
class ResponseRecorder:
    labels: list[str] = field(default_factory=list)

    def capture(self, event: ModelResponse, ctx: Context) -> None:
        self.labels.append(ctx.variables["label"])


def shared_prompt() -> str:
    return "shared"


@dataclass
class PromptUpdater:
    mode: str

    def update(self, event: BaseEvent, ctx: Context) -> None:
        if self.mode == "append":
            ctx.prompt.extend(["persistent", "shared"])
        elif self.mode == "copy":
            ctx.prompt = [*ctx.prompt, "persistent", "shared"]
        elif self.mode == "deepcopy":
            ctx.prompt = [*deepcopy(ctx.prompt), "persistent", "shared"]
        elif self.mode == "copy_fragments":
            ctx.prompt = [*(copy(fragment) for fragment in ctx.prompt), "persistent", "shared"]
        elif self.mode == "replace":
            ctx.prompt = ["persistent", "shared"]
        else:
            ctx.prompt.clear()


@dataclass
class DefaultsUpdater:
    replace_mappings: bool

    def update(self, event: BaseEvent, ctx: Context) -> None:
        if self.replace_mappings:
            ctx.variables = dict(ctx.variables)
            ctx.dependencies = dict(ctx.dependencies)
        ctx.variables["label"] = "updated"
        ctx.dependencies["source"] = "updated"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["append", "copy", "deepcopy", "copy_fragments", "replace", "clear"])
async def test_cleanup_preserves_subscriber_prompt_updates(mode: str) -> None:
    config = RecordingConfig("done", "next")
    agent = Agent("a", prompt="shared", config=config)
    stream = MemoryStream()
    updater = PromptUpdater(mode)
    with stream.where(ModelResponse).sub_scope(updater.update):
        reply = await agent.ask("go", stream=stream, plugins=[Plugin(prompt=["shared", shared_prompt])])
    expected = {
        "append": ["shared", "persistent", "shared"],
        "copy": ["shared", "persistent", "shared"],
        "deepcopy": ["shared", "persistent", "shared"],
        "copy_fragments": ["shared", "persistent", "shared"],
        "replace": ["persistent", "shared"],
        "clear": [],
    }[mode]
    assert config.calls[0].prompt == ["shared", "shared", "shared"]
    assert reply.context.prompt == expected
    await reply.ask("again")
    assert config.calls[-1].prompt == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("replace_mappings", [False, True])
async def test_cleanup_preserves_reassigned_defaults(replace_mappings: bool) -> None:
    config = RecordingConfig("done", "next")
    agent = Agent("a", config=config)
    stream = MemoryStream()
    updater = DefaultsUpdater(replace_mappings)
    plugin = Plugin(
        variables={"label": "plugin", "temporary": "remove"},
        dependencies={"source": "plugin", "temporary": "remove"},
    )
    with stream.where(ModelResponse).sub_scope(updater.update):
        reply = await agent.ask("go", stream=stream, plugins=[plugin])
    assert reply.context.variables == {"label": "updated"}
    assert reply.context.dependencies == IsPartialDict({"source": "updated"})
    assert "temporary" not in reply.context.dependencies
    await reply.ask("again")
    assert config.calls[-1].variables == {"label": "updated"}
    assert config.calls[-1].dependencies == IsPartialDict({"source": "updated"})


@pytest.mark.asyncio
async def test_failure_preserves_subscriber_updates() -> None:
    agent = Agent("a", prompt="shared", config=RecordingConfig("initial"))
    reply = await agent.ask("initial")
    prompt_updater = PromptUpdater("append")
    defaults_updater = DefaultsUpdater(False)
    stream = reply.context.stream
    plugin = Plugin(prompt="shared", variables={"label": "plugin"}, dependencies={"source": "plugin"})
    with (
        stream.where(ModelRequest).sub_scope(prompt_updater.update),
        stream.where(ModelRequest).sub_scope(defaults_updater.update),
        pytest.raises(RuntimeError, match="failure"),
    ):
        await reply.ask("go", config=RecordingConfig(RuntimeError("failure")), plugins=[plugin])
    assert reply.context.prompt == ["shared", "persistent", "shared"]
    assert reply.context.variables == {"label": "updated"}
    assert reply.context.dependencies == IsPartialDict({"source": "updated"})


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run", "resume", "reply_ask", "reply_run"])
async def test_plugins_are_scoped_to_every_invocation(entry: str) -> None:
    config = RecordingConfig("first", ToolCallEvent(name="read_defaults", arguments="{}"), "done", "next")
    agent = Agent("a", prompt="base", config=config)
    plugin = Plugin(
        prompt=["plugin", dynamic_prompt],
        tools=[read_defaults],
        variables={"label": "one"},
        dependencies={"source": "plugin"},
    )
    if entry.startswith("reply_"):
        first = await agent.ask("initial")
        target = first
    else:
        target = agent
        config.events = (ToolCallEvent(name="read_defaults", arguments="{}"), "done", "next")
    if entry.endswith("run"):
        async with target.run("go", plugins=(p for p in [plugin])) as run:
            reply = await run.result()
    elif entry == "resume":
        reply = await agent.resume(ModelRequest.ensure_request(["go"]), plugins=[plugin])
    else:
        reply = await target.ask("go", plugins=[plugin])
    calls = [c for c in config.calls if "read_defaults" in c.names]
    assert calls[0].prompt == ["base", "plugin", "dynamic:one:plugin"]
    assert tool_text(config) == "one:plugin"
    assert reply.context.prompt == ["base"]
    assert "source" not in reply.context.dependencies
    assert "label" not in reply.context.variables
    assert reply.context.variables["tool_result"] == "kept"
    await reply.ask("again")
    assert config.calls[-1].prompt == ["base"]
    assert "read_defaults" not in config.calls[-1].names
    assert agent.system_prompt == ("base",)
    assert agent.tools == []
    assert dict(agent.variables) == {}


@pytest.mark.asyncio
async def test_explicit_prompt_and_defaults_take_precedence() -> None:
    config = RecordingConfig("done")
    agent = Agent("a", prompt="base", config=config, variables={"label": "agent"}, dependencies={"source": "agent"})
    plugin = Plugin(prompt=["plugin", dynamic_prompt], variables={"label": "plugin"}, dependencies={"source": "plugin"})
    reply = await agent.ask(
        "go", prompt=["override"], plugins=[plugin], variables={"label": "call"}, dependencies={"source": "call"}
    )
    assert config.calls[0].prompt == ["override", "plugin", "dynamic:call:call"]
    assert reply.context.prompt == ["override"]
    assert reply.context.variables["label"] == "call"
    assert reply.context.dependencies["source"] == "call"


@pytest.mark.asyncio
async def test_agent_dynamic_prompt_can_use_plugin_defaults() -> None:
    config = RecordingConfig("done")
    agent = Agent("a", prompt=["base", dynamic_prompt], config=config)
    await agent.ask("go", plugins=[Plugin(variables={"label": "one"}, dependencies={"source": "plugin"})])
    assert config.calls[0].prompt == ["base", "dynamic:one:plugin"]


@pytest.mark.asyncio
async def test_multiple_plugins_compose_in_order() -> None:
    config = RecordingConfig("done")
    agent = Agent("a", prompt="base", config=config)
    first = Plugin(prompt="first", variables={"label": "first"}, dependencies={"source": "first"})
    second = Plugin(prompt=["second", dynamic_prompt], variables={"label": "second"}, dependencies={"source": "second"})
    await agent.ask("go", plugins=[first, second])
    assert config.calls[0].prompt == ["base", "first", "second", "dynamic:second:second"]


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_share_plugin_state() -> None:
    config = RecordingConfig("done", gate=asyncio.Event(), rendezvous=True)
    agent = Agent("a", prompt="base", config=config)
    replies = await asyncio.gather(
        *(
            agent.ask(
                label,
                plugins=[Plugin(prompt=dynamic_prompt, variables={"label": label}, dependencies={"source": label})],
            )
            for label in ["one", "two"]
        )
    )
    assert {tuple(c.prompt) for c in config.calls} == {("base", "dynamic:one:one"), ("base", "dynamic:two:two")}
    assert all(r.context.prompt == ["base"] for r in replies)
    assert dict(agent.dependencies) == {}
    assert dict(agent.variables) == {}


@pytest.mark.asyncio
async def test_middleware_observers_and_policies_are_local_to_the_turn() -> None:
    config = RecordingConfig("done")
    recorder = ResponseRecorder()
    stream = MemoryStream()
    agent = Agent("a", prompt="base", config=config, assembly=[AppendPolicy("agent-policy")])
    plugin = Plugin(
        prompt="plugin",
        variables={"label": "one"},
        middleware=[Middleware(SuffixMiddleware)],
        observers=[observer(ModelResponse)(recorder.capture)],
    )
    plugin.add_policy(AppendPolicy("plugin-policy"))
    await agent.ask("go", stream=stream, plugins=[plugin])
    assert config.calls[0].prompt == ["base", "plugin", "agent-policy", "plugin-policy", "middleware"]
    assert recorder.labels == ["one"]
    await agent.ask("again", stream=stream)
    assert config.calls[-1].prompt == ["base", "agent-policy"]
    assert recorder.labels == ["one"]
    assert [p.name for p in agent.assembly] == ["agent-policy"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_hook,call_hook,expected",
    [
        (None, None, "plugin"),
        (agent_answer, None, "agent"),
        (agent_answer, call_answer, "call"),
        (None, call_answer, "call"),
    ],
)
async def test_hitl_precedence(base_hook: Any, call_hook: Any, expected: str) -> None:
    config = RecordingConfig(ToolCallEvent(name="request_input", arguments="{}"), "done")
    agent = Agent("a", config=config, hitl_hook=base_hook)
    await agent.ask("go", plugins=[Plugin(tools=[request_input], hitl_hook=plugin_answer)], hitl_hook=call_hook)
    assert tool_text(config) == expected


@pytest.mark.asyncio
async def test_plugin_only_policies_are_applied() -> None:
    config = RecordingConfig("done")
    agent = Agent("a", config=config)
    plugin = Plugin()
    plugin.add_policy(AppendPolicy("plugin-policy"))
    await agent.ask("go", plugins=[plugin])
    assert config.calls[0].prompt == ["plugin-policy"]
    await agent.ask("again")
    assert config.calls[-1].prompt == []


@pytest.mark.asyncio
async def test_failure_cleans_plugin_context_and_tools() -> None:
    config = RecordingConfig("initial", RuntimeError("failure"), "next")
    agent = Agent("a", prompt="base", config=config)
    first = await agent.ask("initial")
    with pytest.raises(RuntimeError, match="failure"):
        await first.ask("go", plugins=[Plugin(prompt="plugin", variables={"label": "one"})])
    assert first.context.prompt == ["base"]
    assert "label" not in first.context.variables
    await first.ask("again")
    assert config.calls[-1].prompt == ["base"]


@pytest.mark.asyncio
async def test_cancellation_cleans_only_plugin_contributions() -> None:
    gate = asyncio.Event()
    config = RecordingConfig("first")
    agent = Agent("a", prompt="base", config=config)
    reply = await agent.ask("initial")
    config.gate = gate
    config.started.clear()
    async with reply.run(
        "go", config=config, plugins=[Plugin(prompt="plugin", variables={"label": "one", "temporary": "remove"})]
    ) as run:
        task = asyncio.create_task(run.result())
        await config.started.wait()
        reply.context.prompt.append("persistent")
        reply.context.variables["label"] = "updated"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert reply.context.prompt == ["base", "persistent"]
    assert reply.context.variables == {"label": "updated"}


@pytest.mark.asyncio
async def test_tool_override_is_local_and_explicit_tools_win() -> None:
    config = RecordingConfig(ToolCallEvent(name="answer", arguments="{}"), "done")
    agent = Agent("a", config=config, tools=[tool(lambda: "base", name="answer")])
    plugin = Plugin(tools=[tool(lambda: "plugin", name="answer")])
    await agent.ask("go", plugins=[plugin])
    assert tool_text(config) == "plugin"
    await agent.ask("go", plugins=[plugin], tools=[tool(lambda: "explicit", name="answer")])
    assert tool_text(config) == "explicit"
    await agent.ask("go")
    assert tool_text(config) == "base"


@pytest.mark.asyncio
async def test_fresh_skill_plugin_updates_catalog_and_name_schema(tmp_path: Path) -> None:
    runtime = LocalRuntime(dir=tmp_path)
    config = RecordingConfig("done")
    agent = Agent("a", prompt="base", config=config)
    await agent.ask("go", plugins=[SkillPlugin(runtime)])
    assert config.calls[-1].names == []
    skill = tmp_path / "new-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: new-skill\ndescription: Newly installed\n---\nInstructions\n")
    runtime.invalidate()
    config.events = (ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "new-skill"})), "done")
    await agent.ask("go", plugins=[SkillPlugin(runtime)])
    assert "<name>new-skill</name>" in config.calls[-1].prompt[-1]
    assert "Instructions" in tool_text(config)
    schema = next(s for s in config.calls[-1].schemas if isinstance(s, FunctionToolSchema))
    assert asdict(schema) == IsPartialDict({
        "function": IsPartialDict({
            "name": "load_skill",
            "parameters": IsPartialDict({
                "properties": IsPartialDict({"name": IsPartialDict({"const": "new-skill"})}),
            }),
        }),
    })
    with pytest.raises(ToolNotFoundError, match="load_skill"):
        await agent.ask("go")


def fail_prompt(ctx: Context) -> str:
    raise ValueError("invalid prompt")


async def drive_run(run: AgentRun) -> None:
    async with run:
        await run.result()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["ask", "run"])
async def test_queued_continuation_preserves_explicit_overrides(entry: str) -> None:
    agent = Agent("a", prompt="base", config=RecordingConfig("initial"))
    reply = await agent.ask("initial")
    first = RecordingConfig("one", gate=asyncio.Event())
    second = RecordingConfig("two")
    one = asyncio.create_task(
        reply.ask(
            "one",
            config=first,
            plugins=[Plugin(prompt="plugin-one", variables={"label": "one"}, dependencies={"source": "one"})],
        )
    )
    await first.started.wait()
    options: dict[str, Any] = {
        "config": second,
        "prompt": ["override-two"],
        "variables": {"label": "explicit-two"},
        "dependencies": {"source": "explicit-two"},
        "plugins": [
            Plugin(
                prompt=["plugin-two", dynamic_prompt],
                variables={"label": "plugin-two"},
                dependencies={"source": "plugin-two"},
            )
        ],
    }
    if entry == "run":
        two = asyncio.create_task(drive_run(reply.run("two", **options)))
    else:
        two = asyncio.create_task(reply.ask("two", **options))
    await asyncio.sleep(0)
    active_prompt = list(reply.context.prompt)
    active_label = reply.context.variables.get("label")
    active_source = reply.context.dependencies.get("source")
    first.gate.set()
    await asyncio.gather(one, two)
    assert (active_prompt, active_label, active_source) == (["base", "plugin-one"], "one", "one")
    assert second.calls[0].prompt == ["override-two", "plugin-two", "dynamic:explicit-two:explicit-two"]
    assert second.calls[0].variables == {"label": "explicit-two"}
    assert second.calls[0].dependencies == IsPartialDict({"source": "explicit-two"})
    assert reply.context.prompt == ["override-two"]
    assert reply.context.variables == {"label": "explicit-two"}
    assert reply.context.dependencies == IsPartialDict({"source": "explicit-two"})


@pytest.mark.asyncio
async def test_plugin_policies_preserve_existing_middleware_order() -> None:
    config = RecordingConfig("done")
    agent = Agent("a", prompt="base", config=config, assembly=[AppendPolicy("agent-policy")])
    agent.add_middleware(Middleware(SuffixMiddleware))
    plugin = Plugin()
    plugin.add_policy(AppendPolicy("plugin-policy"))
    await agent.ask("plain")
    await agent.ask("with plugin", plugins=[plugin])
    assert [call.prompt for call in config.calls] == [
        ["base", "agent-policy", "middleware"],
        ["base", "agent-policy", "plugin-policy", "middleware"],
    ]


@pytest.mark.asyncio
async def test_prompt_failure_restores_context() -> None:
    config = RecordingConfig("first", "next")
    agent = Agent("a", prompt="base", config=config)
    reply = await agent.ask("initial")
    with pytest.raises(ValueError, match="invalid prompt"):
        await reply.ask("go", plugins=[Plugin(prompt=["plugin", fail_prompt], variables={"label": "one"})])
    assert reply.context.prompt == ["base"]
    assert "label" not in reply.context.variables
    await reply.ask("again")
    assert config.calls[-1].prompt == ["base"]


@pytest.mark.asyncio
async def test_undriven_run_restores_context() -> None:
    config = RecordingConfig("first", "next")
    agent = Agent("a", prompt="base", config=config)
    reply = await agent.ask("initial")
    async with reply.run("go", plugins=[Plugin(prompt="plugin", variables={"label": "one"})]):
        assert reply.context.prompt == ["base", "plugin"]
    assert len(config.calls) == 1
    assert reply.context.prompt == ["base"]
    assert "label" not in reply.context.variables


@pytest.mark.asyncio
async def test_continuations_on_the_same_context_serialize_plugin_binding() -> None:
    initial = RecordingConfig("first")
    agent = Agent("a", prompt="base", config=initial)
    reply = await agent.ask("initial")
    first = RecordingConfig("one", gate=asyncio.Event())
    second = RecordingConfig("two")
    one = asyncio.create_task(
        reply.ask(
            "one",
            config=first,
            plugins=[Plugin(prompt=dynamic_prompt, variables={"label": "one"}, dependencies={"source": "one"})],
        )
    )
    await first.started.wait()
    two = asyncio.create_task(
        reply.ask(
            "two",
            config=second,
            plugins=[Plugin(prompt=dynamic_prompt, variables={"label": "two"}, dependencies={"source": "two"})],
        )
    )
    first.gate.set()
    await asyncio.gather(one, two)
    assert first.calls[0].prompt == ["base", "dynamic:one:one"]
    assert second.calls[0].prompt == ["base", "dynamic:two:two"]
    assert reply.context.prompt == ["base"]
    assert "label" not in reply.context.variables


@pytest.mark.asyncio
async def test_first_plugin_hitl_hook_wins() -> None:
    config = RecordingConfig(ToolCallEvent(name="request_input", arguments="{}"), "done")
    agent = Agent("a", config=config)
    with pytest.warns(UserWarning, match="first wins"):
        await agent.ask(
            "go", plugins=[Plugin(tools=[request_input], hitl_hook=plugin_answer), Plugin(hitl_hook=call_answer)]
        )
    assert tool_text(config) == "plugin"
