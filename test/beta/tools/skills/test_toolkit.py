# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import pytest
from dirty_equals import IsPartialDict

from autogen.beta import Context
from autogen.beta.events import ToolCallEvent, ToolErrorEvent
from autogen.beta.tools import SkillsToolkit
from autogen.beta.tools.sandbox import ExecResult, Sandbox
from autogen.beta.tools.sandbox.adapter import ShellAdapter
from autogen.beta.tools.sandbox.local import LocalSandbox
from autogen.beta.tools.skills import LocalRuntime


@pytest.mark.asyncio
async def test_tool_exposes_all_functions(skill_tree: Path, context: Context) -> None:
    tool = SkillsToolkit(runtime=skill_tree)

    schemas = await tool.schemas(context)

    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"list_skills", "load_skill", "read_skill_resource", "run_skill_script"}


@pytest.mark.asyncio
async def test_run_skill_script_schema(skill_tree: Path, context: Context) -> None:
    run_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).run_skill_script()

    [schema] = await run_tool.schemas(context)

    assert asdict(schema) == {
        "type": "function",
        "function": IsPartialDict({
            "name": "run_skill_script",
            "parameters": IsPartialDict({
                "properties": IsPartialDict({
                    "name": IsPartialDict({"type": "string"}),
                    "script": IsPartialDict({"type": "string"}),
                }),
                "required": ["name", "script"],
            }),
        }),
    }


@pytest.mark.asyncio
async def test_name_param_constrained_to_discovered_skills(skill_tree: Path, context: Context) -> None:
    # The activation tools constrain `name` to a Literal enum of discovered
    # skills so the model cannot pass an unknown name (spec Step 4 Tip).
    toolkit = SkillsToolkit(LocalRuntime(dir=skill_tree))

    [load_schema] = await toolkit.load_skill().schemas(context)
    [run_schema] = await toolkit.run_skill_script().schemas(context)

    expected_enum = sorted(["react-best-practices", "markdown-guide"])
    load_name = asdict(load_schema)["function"]["parameters"]["properties"]["name"]
    run_name = asdict(run_schema)["function"]["parameters"]["properties"]["name"]
    assert sorted(load_name["enum"]) == expected_enum
    assert sorted(run_name["enum"]) == expected_enum


@pytest.mark.asyncio
async def test_name_param_falls_back_to_string_when_empty(tmp_path: Path, context: Context) -> None:
    # No skills → a Literal cannot be empty, so `name` stays a plain string.
    toolkit = SkillsToolkit(LocalRuntime(dir=tmp_path / "empty"))

    [load_schema] = await toolkit.load_skill().schemas(context)

    name_prop = asdict(load_schema)["function"]["parameters"]["properties"]["name"]
    assert name_prop["type"] == "string"
    assert "enum" not in name_prop


@pytest.mark.asyncio
async def test_load_skill_wraps_content_with_resources(skill_tree: Path, context: Context) -> None:
    # load_skill returns the body wrapped in identifying tags, the absolute
    # skill directory, and a non-eager listing of bundled resources (spec Step 4).
    load_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).load_skill()

    event = ToolCallEvent(name="load_skill", arguments=json.dumps({"name": "react-best-practices"}))
    result = await load_tool(event, context)

    assert not isinstance(result, ToolErrorEvent)
    content = result.result.parts[0].content
    assert '<skill_content name="react-best-practices">' in content
    assert "React Best Practices" in content
    assert f"Skill directory: {skill_tree / 'react-best-practices'}" in content
    assert "<file>scripts/scaffold.py</file>" in content
    # Body only: the YAML frontmatter is stripped before wrapping (spec Step 4).
    assert "description: Best practices for React development" not in content
    assert "version: 1.2.0" not in content


@pytest.mark.asyncio
async def test_read_skill_resource_returns_content(skill_tree: Path, context: Context) -> None:
    # read_skill_resource reads a bundled file given a path relative to the
    # skill directory (the files listed in <skill_resources>).
    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()

    args = json.dumps({"name": "react-best-practices", "resource": "scripts/scaffold.py"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert not isinstance(result, ToolErrorEvent)
    assert 'print("scaffold")' in result.result.parts[0].content


@pytest.mark.asyncio
async def test_read_skill_resource_rejects_path_traversal(skill_tree: Path, context: Context) -> None:
    # A resource path escaping the skill directory is rejected, not read.
    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()

    args = json.dumps({"name": "react-best-practices", "resource": "../markdown-guide/SKILL.md"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
async def test_run_skill_script_executes(skill_tree: Path) -> None:
    scripts_dir = skill_tree / "react-best-practices" / "scripts"
    env = ShellAdapter(LocalSandbox(path=scripts_dir, cleanup=False))

    # cwd is scripts_dir, so pass just the filename — same as tool.py does
    result = await env.run("python scaffold.py")

    assert "scaffold" in result


class _FakeRemoteSandbox:
    def __init__(self) -> None:
        self.execs: list[Sequence[str]] = []

    @property
    def workdir(self) -> PurePosixPath:
        return PurePosixPath("/workspace")

    @property
    def host_workdir(self) -> None:
        return None

    async def exec(self, argv: Sequence[str], *, env: object = None, timeout: object = None) -> ExecResult:
        self.execs.append(argv)
        return ExecResult(output="ran", exit_code=0)


class _RemoteFactory:
    """A non-local SandboxFactory (no sync fast path) opened per command."""

    def __init__(self) -> None:
        self.sandbox = _FakeRemoteSandbox()

    @asynccontextmanager
    async def open(self, context: object = None) -> AsyncIterator[Sandbox]:
        yield self.sandbox


@pytest.mark.asyncio
async def test_run_skill_script_runs_in_event_loop_with_remote_backend(skill_tree: Path, context: Context) -> None:
    # Regression for finding #5: run_skill_script is async, so it drives a
    # remote backend with `await env.run(...)` inside the agent's own event
    # loop. The old sync path called env.run_sync(), which nests asyncio.run()
    # and raises "active event loop" for a non-local factory.
    factory = _RemoteFactory()
    runtime = LocalRuntime(dir=skill_tree, sandbox=factory)
    run_tool = SkillsToolkit(runtime=runtime).run_skill_script()

    event = ToolCallEvent(
        name="run_skill_script",
        arguments=json.dumps({"name": "react-best-practices", "script": "scaffold.py"}),
    )
    result = await run_tool(event, context)

    assert not isinstance(result, ToolErrorEvent)
    assert factory.sandbox.execs  # the command reached the backend via the async path


@pytest.mark.asyncio
async def test_local_runtime_uses_supplied_sandbox(tmp_path: Path) -> None:
    # A user-supplied Sandbox backend is honoured by shell(); commands run in
    # the sandbox's own workdir rather than the scripts_dir.
    sandbox_dir = tmp_path / "box"
    sandbox_dir.mkdir()
    (sandbox_dir / "marker.txt").write_text("present")

    runtime = LocalRuntime(dir=tmp_path / "skills", sandbox=LocalSandbox(path=sandbox_dir))
    env = runtime.shell(tmp_path / "unused-scripts-dir")
    result = await env.run("cat marker.txt")

    assert "present" in result
