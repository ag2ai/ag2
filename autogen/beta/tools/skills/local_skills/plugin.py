# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable

from autogen.beta.agent import Plugin
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.skills.local_skills.toolkit import SkillsToolkit
from autogen.beta.tools.skills.runtime import SkillRuntime


def SkillPlugin(  # noqa: N802
    runtime: SkillRuntime | str | os.PathLike[str] | None = None,
    *,
    middleware: Iterable[ToolMiddleware] = (),
) -> Plugin:
    """Skills-spec ``Plugin`` that auto-loads skill metadata into the prompt.

    Follows the ``agentskills.io`` progressive-disclosure pattern, but instead
    of exposing a ``list_skills`` tool call, it injects the skill catalog
    (name + description) into the system prompt on agent startup. The model
    discovers what is available without spending a tool round-trip, then:

    1. ``load_skill(name)`` — read the full ``SKILL.md`` on demand.
    2. ``run_skill_script(name, script, args)`` — execute a skill script.

    Default runtime scans ``./.agents/skills`` and ``~/.agents/skills``::

        agent = Agent("a", config=config, plugins=[SkillPlugin()])

    Custom install directory::

        agent = Agent("a", config=config, plugins=[SkillPlugin("./skills")])

    Args:
        runtime: A :class:`SkillRuntime`, or a path to a skills directory.
            ``None`` uses the default :class:`LocalRuntime`.
        middleware: Tool middleware applied to the skill tools.
    """
    toolkit = SkillsToolkit(runtime, middleware=middleware)

    # Closure built once at plugin construction (not per request), so the
    # prompt hook is reused across every turn without re-allocation — same
    # pattern as Agent's subtask tools. The runtime scan happens when the
    # hook runs (per turn), so newly installed skills are picked up.
    async def _skills_prompt() -> str:
        skills = toolkit.discover_skills()
        if not skills:
            return "You have no local skills available."
        catalog = "\n".join(f"- {s['name']}: {s['description']}" for s in skills)
        return f"You have access to the following local skills:\n<skills>\n{catalog}\n</skills>"

    return Plugin(
        tools=(
            toolkit.load_skill(),
            toolkit.run_skill_script(),
        ),
        prompt=_skills_prompt,
    )
