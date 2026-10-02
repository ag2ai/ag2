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
from ag2.events import BaseEvent, HumanMessage, ModelRequest, ModelResponse, ToolCallEvent, ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.middleware import BaseMiddleware, Middleware
from ag2.middleware.base import LLMCall
from ag2.plugin import Plugin
from ag2.testing import ModelCall, TestConfig, TrackingConfig, Turn
from ag2.tools.final import FunctionToolSchema
from ag2.tools.skills import LocalRuntime, SkillPlugin


def tracked(*turns: Turn) -> TrackingConfig:
    return TrackingConfig(TestConfig(*turns))


def prompts(config: TrackingConfig) -> list[tuple[str, ...]]:
    return [call.prompt for call in config.calls]


def tool_names(call: ModelCall) -> list[str]:
    return [s.function.name for s in call.tools if isinstance(s, FunctionToolSchema)]


def tool_text(config: TrackingConfig) -> str:
    event = config.mock.call_args.args[0]
    assert isinstance(event, ToolResultsEvent)
    [result] = event.results
    [part] = result.result.parts
    return part.content


def add_skill(root: Path, name: str, description: str, body: str) -> None:
    skill = root / name
    skill.mkdir()
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")


@dataclass
class Gate:
    """Hold LLM calls until released; ``release_after`` releases once that many calls have arrived."""

    release_after: int | None = None
    arrivals: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)

    async def wait(self) -> None:
        self.arrivals += 1
        self.started.set()
        if self.release_after is not None and self.arrivals >= self.release_after:
            self.released.set()
        await self.released.wait()


class GateMiddleware(BaseMiddleware):
    def __init__(self, event: BaseEvent, context: Context, *, gate: Gate) -> None:
        super().__init__(event, context)
        self.gate = gate

    async def on_llm_call(self, call_next: LLMCall, events: Sequence[BaseEvent], context: Context) -> ModelResponse:
        await self.gate.wait()
        return await call_next(events, context)


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


def dynamic_prompt(ctx: Context) -> str:
    return f"dynamic:{ctx.variables['label']}:{ctx.dependencies['source']}"


def shared_prompt() -> str:
    return "shared"


def read_defaults(ctx: Context) -> str:
    ctx.variables["tool_result"] = "kept"
    return f"{ctx.variables['label']}:{ctx.dependencies['source']}"


async def request_input(ctx: Context) -> str:
    return await ctx.input("Choose", timeout=1)


def fail_prompt(ctx: Context) -> str:
    raise ValueError("invalid prompt")


def plugin_answer() -> HumanMessage:
    return HumanMessage("plugin")


def agent_answer() -> HumanMessage:
    return HumanMessage("agent")


def call_answer() -> HumanMessage:
    return HumanMessage("call")


def append_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt.extend(["persistent", "shared"])


def copy_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt = [*ctx.prompt, "persistent", "shared"]


def deepcopy_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt = [*deepcopy(ctx.prompt), "persistent", "shared"]


def copy_fragments_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt = [*(copy(fragment) for fragment in ctx.prompt), "persistent", "shared"]


def replace_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt = ["persistent", "shared"]


def clear_prompt(event: BaseEvent, ctx: Context) -> None:
    ctx.prompt.clear()


def update_defaults(event: BaseEvent, ctx: Context) -> None:
    ctx.variables["label"] = "updated"
    ctx.dependencies["source"] = "updated"


def replace_mappings_and_update_defaults(event: BaseEvent, ctx: Context) -> None:
    ctx.variables = dict(ctx.variables)
    ctx.dependencies = dict(ctx.dependencies)
    update_defaults(event, ctx)


async def drive_run(run: AgentRun) -> None:
    async with run:
        await run.result()


@pytest.mark.asyncio
class TestCleanup:
    @pytest.mark.parametrize(
        ("update", "expected"),
        [
            (append_prompt, ["shared", "persistent", "shared"]),
            (copy_prompt, ["shared", "persistent", "shared"]),
            (deepcopy_prompt, ["shared", "persistent", "shared"]),
            (copy_fragments_prompt, ["shared", "persistent", "shared"]),
            (replace_prompt, ["persistent", "shared"]),
            (clear_prompt, []),
        ],
    )
    async def test_preserves_subscriber_prompt_updates(self, update: Any, expected: list[str]) -> None:
        config = tracked("done", "next")
        agent = Agent("a", prompt="shared", config=config)
        stream = MemoryStream()
        with stream.where(ModelResponse).sub_scope(update):
            reply = await agent.ask("go", stream=stream, plugins=[Plugin(prompt=["shared", shared_prompt])])
        await reply.ask("again")
        assert reply.context.prompt == expected
        assert prompts(config) == [("shared", "shared", "shared"), tuple(expected)]

    @pytest.mark.parametrize("update", [update_defaults, replace_mappings_and_update_defaults])
    async def test_preserves_reassigned_defaults(self, update: Any) -> None:
        config = tracked("done", "next")
        agent = Agent("a", config=config)
        stream = MemoryStream()
        plugin = Plugin(
            variables={"label": "plugin", "temporary": "remove"},
            dependencies={"source": "plugin", "temporary": "remove"},
        )
        with stream.where(ModelResponse).sub_scope(update):
            reply = await agent.ask("go", stream=stream, plugins=[plugin])
        await reply.ask("again")
        assert reply.context.variables == {"label": "updated"}
        assert reply.context.dependencies == IsPartialDict({"source": "updated"})
        assert "temporary" not in reply.context.dependencies
        assert config.calls[-1].variables == {"label": "updated"}
        assert config.calls[-1].dependencies == IsPartialDict({"source": "updated"})

    async def test_failure_preserves_subscriber_updates(self) -> None:
        agent = Agent("a", prompt="shared", config=tracked("initial"))
        reply = await agent.ask("initial")
        stream = reply.context.stream
        plugin = Plugin(prompt="shared", variables={"label": "plugin"}, dependencies={"source": "plugin"})
        with (
            stream.where(ModelRequest).sub_scope(append_prompt),
            stream.where(ModelRequest).sub_scope(update_defaults),
            pytest.raises(RuntimeError, match="failure"),
        ):
            await reply.ask("go", config=tracked(RuntimeError("failure")), plugins=[plugin])
        assert reply.context.prompt == ["shared", "persistent", "shared"]
        assert reply.context.variables == {"label": "updated"}
        assert reply.context.dependencies == IsPartialDict({"source": "updated"})

    async def test_failure_cleans_plugin_context_and_tools(self) -> None:
        config = tracked("initial", RuntimeError("failure"), "next")
        agent = Agent("a", prompt="base", config=config)
        first = await agent.ask("initial")
        with pytest.raises(RuntimeError, match="failure"):
            await first.ask("go", plugins=[Plugin(prompt="plugin", variables={"label": "one"})])
        assert first.context.prompt == ["base"]
        assert "label" not in first.context.variables
        await first.ask("again")
        assert config.calls[-1].prompt == ("base",)

    async def test_prompt_failure_restores_context(self) -> None:
        config = tracked("first", "next")
        agent = Agent("a", prompt="base", config=config)
        reply = await agent.ask("initial")
        with pytest.raises(ValueError, match="invalid prompt"):
            await reply.ask("go", plugins=[Plugin(prompt=["plugin", fail_prompt], variables={"label": "one"})])
        assert reply.context.prompt == ["base"]
        assert "label" not in reply.context.variables
        await reply.ask("again")
        assert config.calls[-1].prompt == ("base",)

    async def test_cancellation_cleans_only_plugin_contributions(self) -> None:
        agent = Agent("a", prompt="base", config=tracked("first"))
        reply = await agent.ask("initial")
        gate = Gate()
        plugin = Plugin(prompt="plugin", variables={"label": "one", "temporary": "remove"})
        async with reply.run(
            "go", config=tracked("second"), plugins=[plugin], middleware=[Middleware(GateMiddleware, gate=gate)]
        ) as run:
            task = asyncio.create_task(run.result())
            await gate.started.wait()
            reply.context.prompt.append("persistent")
            reply.context.variables["label"] = "updated"
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert reply.context.prompt == ["base", "persistent"]
        assert reply.context.variables == {"label": "updated"}

    async def test_undriven_run_restores_context(self) -> None:
        config = tracked("first", "next")
        agent = Agent("a", prompt="base", config=config)
        reply = await agent.ask("initial")
        async with reply.run("go", plugins=[Plugin(prompt="plugin", variables={"label": "one"})]):
            assert reply.context.prompt == ["base", "plugin"]
        assert len(config.calls) == 1
        assert reply.context.prompt == ["base"]
        assert "label" not in reply.context.variables


@pytest.mark.asyncio
class TestScope:
    @pytest.mark.parametrize("entry", ["ask", "run", "resume", "reply_ask", "reply_run"])
    async def test_plugins_are_scoped_to_every_invocation(self, entry: str) -> None:
        turns: list[Turn] = [ToolCallEvent(name="read_defaults", arguments="{}"), "done", "next"]
        if entry.startswith("reply_"):
            turns.insert(0, "first")
        config = tracked(*turns)
        agent = Agent("a", prompt="base", config=config)
        plugin = Plugin(
            prompt=["plugin", dynamic_prompt],
            tools=[read_defaults],
            variables={"label": "one"},
            dependencies={"source": "plugin"},
        )
        target = await agent.ask("initial") if entry.startswith("reply_") else agent
        if entry.endswith("run"):
            async with target.run("go", plugins=(p for p in [plugin])) as run:
                reply = await run.result()
        elif entry == "resume":
            reply = await agent.resume(ModelRequest.ensure_request(["go"]), plugins=[plugin])
        else:
            reply = await target.ask("go", plugins=[plugin])
        with_plugin = next(call for call in config.calls if "read_defaults" in tool_names(call))
        assert with_plugin.prompt == ("base", "plugin", "dynamic:one:plugin")
        assert tool_text(config) == "one:plugin"
        assert reply.context.prompt == ["base"]
        assert "source" not in reply.context.dependencies
        assert "label" not in reply.context.variables
        assert reply.context.variables["tool_result"] == "kept"
        await reply.ask("again")
        assert config.calls[-1].prompt == ("base",)
        assert "read_defaults" not in tool_names(config.calls[-1])
        assert agent.system_prompt == ("base",)
        assert agent.tools == []
        assert dict(agent.variables) == {}

    async def test_concurrent_calls_do_not_share_plugin_state(self) -> None:
        config = tracked("done")
        agent = Agent("a", prompt="base", config=config)
        gate = Gate(release_after=2)
        replies = await asyncio.gather(
            *(
                agent.ask(
                    label,
                    plugins=[Plugin(prompt=dynamic_prompt, variables={"label": label}, dependencies={"source": label})],
                    middleware=[Middleware(GateMiddleware, gate=gate)],
                )
                for label in ["one", "two"]
            )
        )
        assert set(prompts(config)) == {("base", "dynamic:one:one"), ("base", "dynamic:two:two")}
        assert all(r.context.prompt == ["base"] for r in replies)
        assert dict(agent.dependencies) == {}
        assert dict(agent.variables) == {}

    async def test_middleware_observers_and_policies_are_local_to_the_turn(self) -> None:
        config = tracked("done", "again")
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
        await agent.ask("again", stream=stream)
        assert recorder.labels == ["one"]
        assert prompts(config) == [
            ("base", "plugin", "agent-policy", "plugin-policy", "middleware"),
            ("base", "agent-policy"),
        ]
        assert [p.name for p in agent.assembly] == ["agent-policy"]

    async def test_plugin_only_policies_are_applied(self) -> None:
        config = tracked("done", "again")
        agent = Agent("a", config=config)
        plugin = Plugin()
        plugin.add_policy(AppendPolicy("plugin-policy"))
        await agent.ask("go", plugins=[plugin])
        await agent.ask("again")
        assert prompts(config) == [("plugin-policy",), ()]

    async def test_plugin_policies_preserve_existing_middleware_order(self) -> None:
        config = tracked("done")
        agent = Agent("a", prompt="base", config=config, assembly=[AppendPolicy("agent-policy")])
        agent.add_middleware(Middleware(SuffixMiddleware))
        plugin = Plugin()
        plugin.add_policy(AppendPolicy("plugin-policy"))
        await agent.ask("plain")
        await agent.ask("with plugin", plugins=[plugin])
        assert prompts(config) == [
            ("base", "agent-policy", "middleware"),
            ("base", "agent-policy", "plugin-policy", "middleware"),
        ]


@pytest.mark.asyncio
class TestPrecedence:
    async def test_explicit_prompt_and_defaults_take_precedence(self) -> None:
        config = tracked("done")
        agent = Agent("a", prompt="base", config=config, variables={"label": "agent"}, dependencies={"source": "agent"})
        plugin = Plugin(
            prompt=["plugin", dynamic_prompt], variables={"label": "plugin"}, dependencies={"source": "plugin"}
        )
        reply = await agent.ask(
            "go", prompt=["override"], plugins=[plugin], variables={"label": "call"}, dependencies={"source": "call"}
        )
        assert prompts(config) == [("override", "plugin", "dynamic:call:call")]
        assert reply.context.prompt == ["override"]
        assert reply.context.variables["label"] == "call"
        assert reply.context.dependencies["source"] == "call"

    async def test_agent_dynamic_prompt_can_use_plugin_defaults(self) -> None:
        config = tracked("done")
        agent = Agent("a", prompt=["base", dynamic_prompt], config=config)
        await agent.ask("go", plugins=[Plugin(variables={"label": "one"}, dependencies={"source": "plugin"})])
        assert prompts(config) == [("base", "dynamic:one:plugin")]

    async def test_multiple_plugins_compose_in_order(self) -> None:
        config = tracked("done")
        agent = Agent("a", prompt="base", config=config)
        first = Plugin(prompt="first", variables={"label": "first"}, dependencies={"source": "first"})
        second = Plugin(
            prompt=["second", dynamic_prompt], variables={"label": "second"}, dependencies={"source": "second"}
        )
        await agent.ask("go", plugins=[first, second])
        assert prompts(config) == [("base", "first", "second", "dynamic:second:second")]

    async def test_tool_override_is_local_and_explicit_tools_win(self) -> None:
        config = tracked(ToolCallEvent(name="answer", arguments="{}"), "done")
        agent = Agent("a", config=config, tools=[tool(lambda: "base", name="answer")])
        plugin = Plugin(tools=[tool(lambda: "plugin", name="answer")])
        await agent.ask("go", plugins=[plugin])
        assert tool_text(config) == "plugin"
        await agent.ask("go", plugins=[plugin], tools=[tool(lambda: "explicit", name="answer")])
        assert tool_text(config) == "explicit"
        await agent.ask("go")
        assert tool_text(config) == "base"

    @pytest.mark.parametrize(
        ("base_hook", "call_hook", "expected"),
        [
            (None, None, "plugin"),
            (agent_answer, None, "agent"),
            (agent_answer, call_answer, "call"),
            (None, call_answer, "call"),
        ],
    )
    async def test_hitl_precedence(self, base_hook: Any, call_hook: Any, expected: str) -> None:
        config = tracked(ToolCallEvent(name="request_input", arguments="{}"), "done")
        agent = Agent("a", config=config, hitl_hook=base_hook)
        await agent.ask("go", plugins=[Plugin(tools=[request_input], hitl_hook=plugin_answer)], hitl_hook=call_hook)
        assert tool_text(config) == expected

    async def test_first_plugin_hitl_hook_wins(self) -> None:
        config = tracked(ToolCallEvent(name="request_input", arguments="{}"), "done")
        agent = Agent("a", config=config)
        with pytest.warns(UserWarning, match="first wins"):
            await agent.ask(
                "go", plugins=[Plugin(tools=[request_input], hitl_hook=plugin_answer), Plugin(hitl_hook=call_answer)]
            )
        assert tool_text(config) == "plugin"

    async def test_conflicting_plugin_hitl_hooks_warn_once(self) -> None:
        agent = Agent("a", config=tracked("done"))
        plugins = [Plugin(hitl_hook=plugin_answer), Plugin(hitl_hook=call_answer), Plugin(hitl_hook=agent_answer)]
        with pytest.warns(UserWarning, match="first wins") as record:
            await agent.ask("go", plugins=plugins)
        assert len(record) == 1


@pytest.mark.asyncio
class TestContinuations:
    @pytest.mark.parametrize("entry", ["ask", "run"])
    async def test_queued_continuation_preserves_explicit_overrides(self, entry: str) -> None:
        agent = Agent("a", prompt="base", config=tracked("initial"))
        reply = await agent.ask("initial")
        gate = Gate()
        second = tracked("two")
        one = asyncio.create_task(
            reply.ask(
                "one",
                config=tracked("one"),
                middleware=[Middleware(GateMiddleware, gate=gate)],
                plugins=[Plugin(prompt="plugin-one", variables={"label": "one"}, dependencies={"source": "one"})],
            )
        )
        await gate.started.wait()
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
        assert reply.context.prompt == ["base", "plugin-one"]
        assert reply.context.variables == {"label": "one"}
        assert reply.context.dependencies == IsPartialDict({"source": "one"})
        gate.released.set()
        await asyncio.gather(one, two)
        assert second.calls == [
            ModelCall(
                prompt=("override-two", "plugin-two", "dynamic:explicit-two:explicit-two"),
                tools=(),
                dependencies=IsPartialDict({"source": "explicit-two"}),
                variables={"label": "explicit-two"},
            )
        ]
        assert reply.context.prompt == ["override-two"]
        assert reply.context.variables == {"label": "explicit-two"}
        assert reply.context.dependencies == IsPartialDict({"source": "explicit-two"})

    async def test_continuations_on_the_same_context_serialize_plugin_binding(self) -> None:
        agent = Agent("a", prompt="base", config=tracked("first"))
        reply = await agent.ask("initial")
        gate = Gate()
        first, second = tracked("one"), tracked("two")
        one = asyncio.create_task(
            reply.ask(
                "one",
                config=first,
                middleware=[Middleware(GateMiddleware, gate=gate)],
                plugins=[Plugin(prompt=dynamic_prompt, variables={"label": "one"}, dependencies={"source": "one"})],
            )
        )
        await gate.started.wait()
        two = asyncio.create_task(
            reply.ask(
                "two",
                config=second,
                plugins=[Plugin(prompt=dynamic_prompt, variables={"label": "two"}, dependencies={"source": "two"})],
            )
        )
        gate.released.set()
        await asyncio.gather(one, two)
        assert prompts(first) == [("base", "dynamic:one:one")]
        assert prompts(second) == [("base", "dynamic:two:two")]
        assert reply.context.prompt == ["base"]
        assert "label" not in reply.context.variables


@pytest.mark.asyncio
class TestSkillPlugin:
    async def test_fresh_plugin_updates_catalog_and_name_schema(self, tmp_path: Path) -> None:
        runtime = LocalRuntime(dir=tmp_path)
        before = tracked("done")
        agent = Agent("a", prompt="base", config=before)
        await agent.ask("go", plugins=[SkillPlugin(runtime)])
        assert [tool_names(call) for call in before.calls] == [[]]
        add_skill(tmp_path, "new-skill", "Newly installed", "Instructions")
        runtime.invalidate()
        after = tracked(ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "new-skill"})), "done")
        await agent.ask("go", config=after, plugins=[SkillPlugin(runtime)])
        [catalog] = [p for p in after.calls[-1].prompt if "<available_skills>" in p]
        assert "<name>new-skill</name>" in catalog
        assert "Instructions" in tool_text(after)
        [schema] = [s for s in after.calls[-1].tools if isinstance(s, FunctionToolSchema)]
        assert asdict(schema) == IsPartialDict({
            "function": IsPartialDict({
                "name": "load_skill",
                "parameters": IsPartialDict({
                    "properties": IsPartialDict({"name": IsPartialDict({"const": "new-skill"})}),
                }),
            }),
        })
        with pytest.raises(ToolNotFoundError, match="load_skill"):
            await agent.ask("go", config=tracked(ToolCallEvent(name="load_skill", arguments="{}"), "done"))

    async def test_fresh_plugin_tool_overrides_constructor_plugin(self, tmp_path: Path) -> None:
        runtime = LocalRuntime(dir=tmp_path)
        add_skill(tmp_path, "old-skill", "Old", "Old body")
        agent = Agent("a", config=tracked("done"), plugins=[SkillPlugin(runtime)])
        add_skill(tmp_path, "new-skill", "New", "New body")
        runtime.invalidate()
        config = tracked(ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "new-skill"})), "done")
        await agent.ask("go", config=config, plugins=[SkillPlugin(runtime)])
        assert "New body" in tool_text(config)
