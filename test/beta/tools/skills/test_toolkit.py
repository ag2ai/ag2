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
async def test_tool_exposes_three_functions(skill_tree: Path, context: Context) -> None:
    tool = SkillsToolkit(runtime=skill_tree)

    schemas = await tool.schemas(context)

    assert len(schemas) == 3
    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"list_skills", "load_skill", "run_skill_script"}


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
