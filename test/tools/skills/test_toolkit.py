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

from ag2 import Context
from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.tools import SkillsToolkit
from ag2.tools.sandbox import ExecResult, Sandbox
from ag2.tools.sandbox.adapter import ShellAdapter
from ag2.tools.sandbox.local import LocalSandbox
from ag2.tools.skills import LocalRuntime


def _write_script_skill(base: Path, name: str, script_body: str | None = None) -> Path:
    """Write a skill with an optional ``scripts/go.sh`` and return *base*."""
    skill_dir = base / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {name}\n---\n# {name}\n")
    if script_body is not None:
        (skill_dir / "scripts").mkdir()
        (skill_dir / "scripts" / "go.sh").write_text(script_body)
    return base


def _symlink_or_skip(link: Path, target: Path) -> None:
    """Create ``link`` -> ``target``, skipping the test where symlinks are unsupported."""
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):  # pragma: no cover - e.g. Windows without privilege
        pytest.skip("symlinks are not supported on this platform")


@pytest.mark.asyncio
async def test_run_script_routes_to_last_runtime(tmp_path: Path, context: Context) -> None:
    # On a name clash, execution routes to the last runtime (project > global).
    glob = _write_script_skill(tmp_path / "global", "dup", "echo GLOBAL\n")
    proj = _write_script_skill(tmp_path / "project", "dup", "echo PROJECT\n")
    run_tool = SkillsToolkit(LocalRuntime(dir=glob), LocalRuntime(dir=proj)).run_skill_script()

    args = json.dumps({"name": "dup", "script": "go.sh"})
    result = await run_tool(ToolCallEvent(name="run_skill_script", arguments=args), context)

    assert not isinstance(result, ToolErrorEvent)
    assert "PROJECT" in result.result.parts[0].content
    assert "GLOBAL" not in result.result.parts[0].content


@pytest.mark.asyncio
async def test_run_script_falls_through_on_skill_not_found(tmp_path: Path, context: Context) -> None:
    # The last runtime does not own the skill (SkillNotFoundError), so the chain
    # falls through to the global runtime that does.
    glob = _write_script_skill(tmp_path / "global", "only-global", "echo HELLO\n")
    proj = tmp_path / "project"
    proj.mkdir()
    run_tool = SkillsToolkit(LocalRuntime(dir=glob), LocalRuntime(dir=proj)).run_skill_script()

    args = json.dumps({"name": "only-global", "script": "go.sh"})
    result = await run_tool(ToolCallEvent(name="run_skill_script", arguments=args), context)

    assert not isinstance(result, ToolErrorEvent)
    assert "HELLO" in result.result.parts[0].content


@pytest.mark.asyncio
async def test_missing_script_in_owning_runtime_does_not_fall_through(tmp_path: Path, context: Context) -> None:
    # The last runtime OWNS the skill but lacks the script: that is a genuine
    # error, not SkillNotFoundError, so the chain must NOT fall through and run
    # the global runtime's same-named script.
    glob = _write_script_skill(tmp_path / "global", "dup", "echo GLOBAL\n")
    proj = _write_script_skill(tmp_path / "project", "dup", script_body=None)  # owns dup, no scripts
    run_tool = SkillsToolkit(LocalRuntime(dir=glob), LocalRuntime(dir=proj)).run_skill_script()

    args = json.dumps({"name": "dup", "script": "go.sh"})
    result = await run_tool(ToolCallEvent(name="run_skill_script", arguments=args), context)

    assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
async def test_tool_exposes_all_functions(skill_tree: Path, context: Context) -> None:
    tool = SkillsToolkit(skill_tree)

    schemas = await tool.schemas(context)

    names = {s.function.name for s in schemas}  # type: ignore[union-attr]
    assert names == {"list_skills", "load_skill", "read_skill_resource", "run_skill_script"}


@pytest.mark.asyncio
async def test_run_skill_script_schema(skill_tree: Path, context: Context) -> None:
    run_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).run_skill_script()

    [schema] = await run_tool.schemas(context)

    assert asdict(schema) == {
        "type": "function",
        "defer_loading": False,
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
    assert "<file>references/guide.md</file>" in content
    # Scripts are not resources: scripts/ is excluded from the resource listing.
    assert "<file>scripts/scaffold.py</file>" not in content
    # Body only: the YAML frontmatter is stripped before wrapping (spec Step 4).
    assert "description: Best practices for React development" not in content
    assert "version: 1.2.0" not in content


@pytest.mark.asyncio
async def test_read_skill_resource_returns_content(skill_tree: Path, context: Context) -> None:
    # read_skill_resource reads a bundled file given a path relative to the
    # skill directory (the files listed in <skill_resources>).
    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()

    args = json.dumps({"name": "react-best-practices", "resource": "references/guide.md"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert not isinstance(result, ToolErrorEvent)
    assert "Detailed React guidance." in result.result.parts[0].content


@pytest.mark.asyncio
async def test_read_skill_resource_rejects_path_traversal(skill_tree: Path, context: Context) -> None:
    # A resource path escaping the skill directory is rejected, not read.
    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()

    args = json.dumps({"name": "react-best-practices", "resource": "../markdown-guide/SKILL.md"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
async def test_read_skill_resource_rejects_symlinked_resource_escape(skill_tree: Path, context: Context) -> None:
    # A symlinked resource is *discovered* (Path.is_file follows the link, so it
    # lands in the descriptor list), so the membership check alone passes. The
    # resolved target escapes the skill directory, so the read must still be
    # rejected — otherwise a bundled symlink leaks arbitrary files.
    secret = skill_tree / "secret.txt"  # outside any skill directory
    secret.write_text("TOP SECRET\n")
    _symlink_or_skip(skill_tree / "react-best-practices" / "leak.txt", secret)

    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()
    args = json.dumps({"name": "react-best-practices", "resource": "leak.txt"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert isinstance(result, ToolErrorEvent)


@pytest.mark.asyncio
async def test_read_skill_resource_allows_in_bounds_symlink(skill_tree: Path, context: Context) -> None:
    # The guard rejects *escapes*, not symlinks per se: a symlink whose target
    # resolves back inside the skill directory is still served.
    _symlink_or_skip(
        skill_tree / "react-best-practices" / "alias.md",
        skill_tree / "react-best-practices" / "references" / "guide.md",
    )

    read_tool = SkillsToolkit(LocalRuntime(dir=skill_tree)).read_skill_resource()
    args = json.dumps({"name": "react-best-practices", "resource": "alias.md"})
    result = await read_tool(ToolCallEvent(name="read_skill_resource", arguments=args), context)

    assert not isinstance(result, ToolErrorEvent)
    assert "Detailed React guidance." in result.result.parts[0].content


@pytest.mark.asyncio
async def test_run_skill_script_rejects_symlinked_script_escape(tmp_path: Path, context: Context) -> None:
    # Mirror of the resource escape for execution: a symlinked script under
    # scripts/ is discovered and passes membership, but resolves outside the
    # scripts directory, so it must be rejected rather than executed.
    skills_root = _write_script_skill(tmp_path / "skills", "evil-skill", "echo SAFE\n")
    outside = tmp_path / "outside.sh"
    outside.write_text("echo PWNED\n")
    _symlink_or_skip(skills_root / "evil-skill" / "scripts" / "evil.sh", outside)

    run_tool = SkillsToolkit(LocalRuntime(dir=skills_root)).run_skill_script()
    args = json.dumps({"name": "evil-skill", "script": "evil.sh"})
    result = await run_tool(ToolCallEvent(name="run_skill_script", arguments=args), context)

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
    run_tool = SkillsToolkit(runtime).run_skill_script()

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
