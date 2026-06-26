# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Annotated

import pytest
from pydantic import ValidationError

from ag2 import Agent, Context, Depends, Variable
from ag2.events import ToolCallEvent, ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools.skills import MemoryRuntime, MemorySkill, SkillPlugin


def _call(tool: str, **arguments: object) -> ToolCallEvent:
    return ToolCallEvent(name=tool, arguments=json.dumps(arguments))


def _tool_result(tracking: TrackingConfig, call: int = 1) -> str:
    """Content of the first tool result fed back to the LLM on the *call*-th turn."""
    msg: ToolResultsEvent = tracking.mock.call_args_list[call][0][0]
    return msg.results[0].result.parts[0].content


@pytest.mark.asyncio
class TestCatalog:
    async def test_memory_skill_listed_in_catalog(self) -> None:
        skill = MemorySkill(name="greeter", description="Greets people", instructions="Say hi")
        agent = Agent("a", config=TestConfig("done"), plugins=[SkillPlugin(skill)])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "<name>greeter</name>" in catalog
        assert "<description>Greets people</description>" in catalog

    async def test_load_skill_returns_instructions(self) -> None:
        skill = MemorySkill(name="greeter", description="d", instructions="Say hello warmly.")
        tracking = TrackingConfig(TestConfig(_call("load_skill", name="greeter"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert "Say hello warmly." in _tool_result(tracking)


@pytest.mark.asyncio
class TestScripts:
    async def test_load_skill_embeds_parameters_schema(self) -> None:
        skill = MemorySkill(name="conv", description="d", instructions="Use convert.")

        @skill.script
        def convert(value: float, factor: float) -> str:
            """Multiply value by factor."""
            return str(value * factor)

        tracking = TrackingConfig(TestConfig(_call("load_skill", name="conv"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        content = _tool_result(tracking)
        assert "<scripts>" in content
        # name comes from the function, description from the docstring (no explicit args)
        assert '<script name="convert" description="Multiply value by factor.">' in content
        assert "<parameters_schema>" in content
        assert "factor" in content

    async def test_run_script_serializes_and_coerces_args(self) -> None:
        # Args are coerced to the annotated types via FastDepends, like a tool:
        # the ints 10 and 2 become floats, so the result is 20.0, not 20.
        skill = MemorySkill(name="conv", description="d")

        @skill.script
        def convert(value: float, factor: float) -> str:
            return str(value * factor)

        tracking = TrackingConfig(
            TestConfig(
                _call("run_skill_script", name="conv", script="convert", args={"value": 10, "factor": 2}), "done"
            )
        )
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert "20.0" in _tool_result(tracking)

    async def test_run_async_script(self) -> None:
        skill = MemorySkill(name="conv", description="d")

        @skill.script
        async def echo(text: str) -> str:
            return text.upper()

        tracking = TrackingConfig(
            TestConfig(_call("run_skill_script", name="conv", script="echo", args={"text": "hi"}), "done")
        )
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert "HI" in _tool_result(tracking)

    async def test_invalid_args_raise_validation_error(self) -> None:
        skill = MemorySkill(name="conv", description="d")

        @skill.script
        def convert(value: float, factor: float) -> str:
            return str(value * factor)

        call = _call("run_skill_script", name="conv", script="convert", args={"value": "x", "factor": 2})
        agent = Agent("a", config=TestConfig(call, "done"), plugins=[SkillPlugin(skill)])

        with pytest.raises(ValidationError):
            await agent.ask("hi")

    async def test_array_args_rejected_for_in_process_script(self) -> None:
        skill = MemorySkill(name="conv", description="d")

        @skill.script
        def convert(value: float, factor: float) -> str:
            return str(value * factor)

        call = _call("run_skill_script", name="conv", script="convert", args=["10", "2"])
        agent = Agent("a", config=TestConfig(call, "done"), plugins=[SkillPlugin(skill)])

        with pytest.raises(TypeError, match="requires named arguments"):
            await agent.ask("hi")


@pytest.mark.asyncio
class TestNamingDefaults:
    async def test_name_from_function_and_description_from_docstring(self) -> None:
        skill = MemorySkill(name="s", description="d", instructions="body")

        @skill.resource
        def roster() -> str:
            """The current team roster."""
            return "Alice"

        @skill.script
        def analyze(text: str) -> str:
            """Analyze the given text."""
            return text

        tracking = TrackingConfig(TestConfig(_call("load_skill", name="s"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        content = _tool_result(tracking)
        assert "<file>roster</file>" in content  # resource name defaults to the function name
        assert '<script name="analyze" description="Analyze the given text.">' in content

    async def test_explicit_name_and_description_override(self) -> None:
        skill = MemorySkill(name="s", description="d", instructions="body")

        @skill.script(name="run", description="custom")
        def analyze(text: str) -> str:
            """ignored docstring"""
            return text

        tracking = TrackingConfig(TestConfig(_call("load_skill", name="s"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert '<script name="run" description="custom">' in _tool_result(tracking)


@pytest.mark.asyncio
class TestResources:
    async def test_read_resource_runs_callable(self) -> None:
        skill = MemorySkill(name="proj", description="d")

        @skill.resource
        async def roster() -> str:
            return "Alice, Bob"

        tracking = TrackingConfig(TestConfig(_call("read_skill_resource", name="proj", resource="roster"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert _tool_result(tracking) == "Alice, Bob"

    async def test_resource_listed_in_loaded_content(self) -> None:
        skill = MemorySkill(name="proj", description="d", instructions="body")

        @skill.resource
        def roster() -> str:
            return "x"

        tracking = TrackingConfig(TestConfig(_call("load_skill", name="proj"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi")

        assert "<file>roster</file>" in _tool_result(tracking)


@pytest.mark.asyncio
class TestDependencyInjection:
    async def test_script_resolves_variable(self) -> None:
        skill = MemorySkill(name="s", description="d")

        @skill.script
        def greet(value: int, who: Annotated[str, Variable("who")]) -> str:
            return f"{who}:{value}"

        tracking = TrackingConfig(
            TestConfig(_call("run_skill_script", name="s", script="greet", args={"value": 1}), "done")
        )
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi", variables={"who": "Ada"})

        assert _tool_result(tracking) == "Ada:1"

    async def test_script_receives_context(self) -> None:
        skill = MemorySkill(name="s", description="d")

        @skill.script
        def peek(ctx: Context) -> str:  # type: ignore[valid-type]
            return ctx.variables["k"]

        tracking = TrackingConfig(TestConfig(_call("run_skill_script", name="s", script="peek"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi", variables={"k": "v"})

        assert _tool_result(tracking) == "v"

    async def test_resource_resolves_variable(self) -> None:
        skill = MemorySkill(name="s", description="d")

        @skill.resource
        def where(region: Annotated[str, Variable("region")]) -> str:
            return region

        tracking = TrackingConfig(TestConfig(_call("read_skill_resource", name="s", resource="where"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])

        await agent.ask("hi", variables={"region": "eu-west-1"})

        assert _tool_result(tracking) == "eu-west-1"

    async def test_script_resolves_depends_via_agent_provider(self) -> None:
        # The agent's dependency provider flows through context.dependency_provider,
        # so a script's `Depends` resolves (and overrides apply) like a tool's.
        def dep1() -> str:
            raise ValueError("not overridden")

        def dep2() -> str:
            return "from-provider"

        skill = MemorySkill(name="s", description="d")

        @skill.script
        def use(dep: Annotated[str, Depends(dep1)]) -> str:
            return dep

        tracking = TrackingConfig(TestConfig(_call("run_skill_script", name="s", script="use"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(skill)])
        agent.dependency_provider.override(dep1, dep2)

        await agent.ask("hi")

        assert _tool_result(tracking) == "from-provider"


@pytest.mark.asyncio
class TestComposition:
    async def test_loose_memory_skills_compose(self) -> None:
        a = MemorySkill(name="a", description="skill a")
        b = MemorySkill(name="b", description="skill b")
        agent = Agent("x", config=TestConfig("done"), plugins=[SkillPlugin(a, b)])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "<name>a</name>" in catalog
        assert "<name>b</name>" in catalog

    async def test_grouping_into_runtimes_is_equivalent(self) -> None:
        # Flat, one-runtime-each, and grouped all yield the same catalog.
        flat = SkillPlugin(MemorySkill(name="a", description="d"), MemorySkill(name="b", description="d2"))
        wrapped = SkillPlugin(
            MemoryRuntime(MemorySkill(name="a", description="d")),
            MemoryRuntime(MemorySkill(name="b", description="d2")),
        )
        grouped = SkillPlugin(
            MemoryRuntime(MemorySkill(name="a", description="d"), MemorySkill(name="b", description="d2"))
        )

        catalogs = []
        for plugin in (flat, wrapped, grouped):
            reply = await Agent("x", config=TestConfig("done"), plugins=[plugin]).ask("hi")
            catalogs.append(reply.context.prompt)

        assert catalogs[0] == catalogs[1] == catalogs[2]
        assert "<name>a</name>" in catalogs[0][0]
        assert "<name>b</name>" in catalogs[0][0]

    async def test_memory_skill_shadows_local_when_declared_last(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "dup"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: dup\ndescription: local\n---\n# local body\n")
        mem = MemorySkill(name="dup", description="memory", instructions="memory body")

        tracking = TrackingConfig(TestConfig(_call("load_skill", name="dup"), "done"))
        agent = Agent("a", config=tracking, plugins=[SkillPlugin(tmp_path, mem)])

        reply = await agent.ask("hi")

        [catalog] = reply.context.prompt
        assert "<description>memory</description>" in catalog
        assert "memory body" in _tool_result(tracking)

    async def test_script_tool_gated_out_when_no_scripts(self) -> None:
        skill = MemorySkill(name="s", description="d", instructions="body")
        agent = Agent(
            "a", config=TestConfig(ToolCallEvent(name="run_skill_script"), "done"), plugins=[SkillPlugin(skill)]
        )

        with pytest.raises(ToolNotFoundError, match="run_skill_script"):
            await agent.ask("hi")

    async def test_resource_tool_gated_out_when_no_resources(self) -> None:
        skill = MemorySkill(name="s", description="d")

        @skill.script
        def go() -> str:
            return "ok"

        agent = Agent(
            "a", config=TestConfig(ToolCallEvent(name="read_skill_resource"), "done"), plugins=[SkillPlugin(skill)]
        )

        with pytest.raises(ToolNotFoundError, match="read_skill_resource"):
            await agent.ask("hi")


class TestMemoryRuntimeReadOnly:
    # The read-only contract is not reachable through Agent — it is exercised only
    # by the skill install flow (SkillSearchToolkit), so these stay direct.
    def test_cleanup_is_false(self) -> None:
        assert MemoryRuntime(MemorySkill(name="a", description="d")).cleanup is False

    def test_ensure_storage_is_noop(self) -> None:
        MemoryRuntime(MemorySkill(name="a", description="d")).ensure_storage()  # does not raise

    def test_install_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            MemoryRuntime(MemorySkill(name="a", description="d")).install(tmp_path, "a")

    def test_remove_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            MemoryRuntime(MemorySkill(name="a", description="d")).remove("a")

    def test_lock_dir_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _ = MemoryRuntime(MemorySkill(name="a", description="d")).lock_dir
