# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Annotated

import pytest

from ag2 import Agent, Context, MemoryStream, Scope, ScopeError, Var, Variable
from ag2.events import ModelMessage, ModelResponse, ToolCallEvent, Usage
from ag2.testing import TestConfig
from ag2.tools.subagents.run_task import run_task


async def contribute(who: str, ctx: Context) -> str:
    """Contribute to a shared variable while updating independent copied values."""
    assert "secret" not in ctx.variables
    assert "ag:private" not in ctx.variables
    async with ctx.dependencies["findings_lock"]:
        findings = ctx.variables["findings"]
        findings.append(who)
        ctx.variables["findings"] = [*findings, f"{who}-rebound"]
    ctx.variables["depth"] += 1
    ctx.variables["nested"]["n"] += 1
    return str(ctx.variables["depth"])


def collector(name: str) -> Agent:
    return Agent(
        name,
        config=TestConfig(
            ToolCallEvent(name="contribute", arguments=f'{{"who": "{name}"}}'),
            "done",
        ),
        tools=[contribute],
    )


@pytest.mark.asyncio
async def test_parallel_children_share_rebindings_and_isolate_copied_and_local() -> None:
    alpha, beta = collector("alpha"), collector("beta")
    root = Agent(
        "root",
        config=TestConfig(
            [
                ToolCallEvent(name="task_alpha", arguments='{"objective": "contribute"}'),
                ToolCallEvent(name="task_beta", arguments='{"objective": "contribute"}'),
            ],
            "done",
        ),
        tools=[alpha.as_tool(description="Contribute"), beta.as_tool(description="Contribute")],
        dependencies={"findings_lock": asyncio.Lock()},
        variables={
            "findings": Var([], scope=Scope.SHARED),
            "depth": 0,
            "nested": {"n": 0},
            "secret": Var("root only", scope=Scope.LOCAL),
            "ag:private": "control",
        },
    )
    reply = await root.ask("start")
    assert sorted(reply.context.variables["findings"]) == ["alpha", "alpha-rebound", "beta", "beta-rebound"]
    assert reply.context.variables["nested"] == {"n": 0}
    assert reply.context.variables["depth"] == 0
    assert reply.context.variables["secret"] == "root only"


async def inject_findings(findings: Annotated[list[str], Variable("findings")]) -> str:
    """Mutate the directly injected shared value, including an initially empty list."""
    findings.append("injected")
    return "done"


@pytest.mark.asyncio
async def test_variable_injection_returns_the_mutable_shared_value() -> None:
    child = Agent(
        "child",
        config=TestConfig(ToolCallEvent(name="inject_findings", arguments="{}"), "done"),
        tools=[inject_findings],
    )
    root = Agent(
        "root",
        config=TestConfig(ToolCallEvent(name="task_child", arguments='{"objective":"work"}'), "done"),
        tools=[child.as_tool(description="Work")],
        variables={"findings": Var([], scope=Scope.SHARED)},
    )
    reply = await root.ask("start")
    assert reply.context.variables["findings"] == ["injected"]


@pytest.mark.asyncio
async def test_child_cannot_introduce_a_new_shared_name() -> None:
    child = Agent("child", config=TestConfig("unused"), variables={"new": Var({}, scope=Scope.SHARED)})
    result = await run_task(child, "work", parent_context=Context(MemoryStream()))
    assert not result.completed
    assert isinstance(result.error, ScopeError)


@pytest.mark.asyncio
async def test_equal_cost_calls_remain_distinct_and_roots_do_not_share_defaults() -> None:
    child = Agent(
        "child",
        config=TestConfig(ModelResponse(tool_calls=None, message=ModelMessage("done"), usage=Usage(total_tokens=10))),
    )
    root = Agent(
        "root",
        config=TestConfig(
            [
                ToolCallEvent(name="task_child", arguments='{"objective":"one"}'),
                ToolCallEvent(name="task_child", arguments='{"objective":"two"}'),
            ],
            "done",
        ),
        tools=[child.as_tool(description="Work")],
        variables={"kv": Var({}, scope=Scope.SHARED)},
    )
    first = await root.ask("start")
    first.context.variables["kv"]["first"] = 1
    second = await root.ask("start")
    assert second.context.variables["kv"] == {}
    assert (await first.usage()).total.total_tokens == 20
    assert (await second.usage()).total.total_tokens == 20
