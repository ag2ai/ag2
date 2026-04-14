# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import shlex
import stat
from pathlib import Path
from typing import Annotated

from pydantic import Field

from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.runtime.local import LocalRuntime
from autogen.beta.tools.runtime.protocol import SkillRuntime


class LocalSkillsTool(Toolkit):
    """Client-side skills toolkit using the agentskills.io convention.

    Implements a three-step progressive-disclosure pattern:

    1. **list_skills()** — returns a lightweight catalog (name + description).
    2. **load_skill(name)** — returns the full ``SKILL.md`` instructions on demand.
    3. **run_skill_script(name, script, args)** — executes a script from the
       skill's ``scripts/`` directory.

    Works with *any* provider (no provider-specific API required).

    Example::
        # Default runtime (.agents/skills + ~/.agents/skills)
        LocalSkillsTool()

        # Custom install directory
        LocalSkillsTool(runtime=LocalRuntime("./skills"))

        # Additional read-only search paths
        LocalSkillsTool(runtime=LocalRuntime("./skills", extra_paths=["./shared-skills"]))
    """

    list_skills: FunctionTool
    load_skill: FunctionTool
    run_skill_script: FunctionTool

    def __init__(
        self,
        runtime: SkillRuntime | None = None,
    ) -> None:
        _runtime: SkillRuntime = runtime if runtime is not None else LocalRuntime()
        self.list_skills = _make_list_tool(_runtime)
        self.load_skill = _make_load_tool(_runtime)
        self.run_skill_script = _make_run_tool(_runtime)

        super().__init__(self.list_skills, self.load_skill, self.run_skill_script)


def _make_list_tool(runtime: SkillRuntime) -> FunctionTool:
    @tool(description="List available local skills with name and short description.")
    def list_skills() -> list[dict[str, str]]:
        return [{"name": m.name, "description": m.description} for m in runtime.discover()]

    return list_skills


def _make_load_tool(runtime: SkillRuntime) -> FunctionTool:
    @tool(description="Load the full SKILL.md content for a specific skill.")
    def load_skill(
        name: Annotated[str, Field(description="Skill name returned by list_skills.")],
    ) -> str:
        return runtime.load(name)

    return load_skill


def _make_run_tool(runtime: SkillRuntime) -> FunctionTool:
    @tool(description="Run a script from a skill's scripts directory. Only .py and .sh scripts are supported.")
    def run_skill_script(
        name: Annotated[str, Field(description="Skill name returned by list_skills.")],
        script: Annotated[
            str,
            Field(description="Script filename inside scripts/, for example scaffold.py or build.sh."),
        ],
        args: Annotated[
            list[str] | None,
            Field(description="Optional script arguments passed as positional parameters."),
        ] = None,
    ) -> str:
        skill_dir = runtime.get_path(name)
        scripts_dir = skill_dir / "scripts"
        script_path = Path(script)
        if script_path.name != script:
            raise ValueError("script must be a filename inside the skill scripts directory")

        resolved_script = (scripts_dir / script_path.name).resolve()
        if not resolved_script.is_file() or not resolved_script.is_relative_to(scripts_dir.resolve()):
            raise FileNotFoundError(f"script {script!r} not found in {scripts_dir}")

        first_line = resolved_script.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
        has_shebang = first_line.startswith("#!")

        if has_shebang:
            resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            command = [f"./{resolved_script.name}"]
        elif resolved_script.suffix.lower() == ".py":
            command = ["python3", f"./{resolved_script.name}"]
        elif resolved_script.suffix.lower() == ".sh":
            command = ["sh", f"./{resolved_script.name}"]
        else:
            resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            command = [f"./{resolved_script.name}"]

        if args:
            command.extend(args)

        env = runtime.shell(scripts_dir)
        return env.run(shlex.join(command))

    return run_skill_script
