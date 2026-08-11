# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from xml.sax.saxutils import escape

from ag2.middleware import ToolMiddleware
from ag2.plugin import Plugin
from ag2.tools.skills.runtime import MemorySkill, SkillRuntime
from ag2.tools.skills.skill_types import Skill
from ag2.tools.skills.toolkit import SkillsToolkit


def SkillPlugin(  # noqa: N802
    *runtimes: SkillRuntime | str | os.PathLike[str] | MemorySkill,
    middleware: Iterable[ToolMiddleware] = (),
) -> Plugin:
    """Skills-spec ``Plugin`` that auto-loads skill metadata into the prompt.

    Follows the ``agentskills.io`` progressive-disclosure pattern, but instead
    of exposing a ``list_skills`` tool call, it injects the skill catalog as an
    ``<available_skills>`` XML block (name + description + location per skill)
    into the system prompt on agent startup. The model discovers what is
    available without spending a tool round-trip, then:

    1. ``load_skill(name)`` — read the full ``SKILL.md`` on demand.
    2. ``read_skill_resource(name, resource)`` — read a bundled resource file.
    3. ``run_skill_script(name, script, args)`` — execute a skill script.

    Accepts **one or more** runtimes (a path is wrapped in a
    :class:`LocalRuntime`), composing them so global and project skills can be
    served at once, each with its own execution/storage config::

        SkillPlugin(LocalRuntime("~/.agents/skills"), LocalRuntime(".agents/skills"))

    On a name clash the **last** runtime wins (project overrides global), applied
    uniformly to the catalog, the activation tools, and routing.

    The catalog and the activation tools are a **construction-time snapshot**:
    the tools constrain ``name`` to the merged set of skills present when the
    plugin is built, and the catalog lists exactly that set — the two never drift
    apart. When no skills are found, the plugin contributes nothing (no catalog,
    no tools). The activation tools are **gated on capability**:
    ``read_skill_resource`` is registered only when some skill has resources, and
    ``run_skill_script`` only when some skill has scripts — so the model is never
    shown a dead tool.

    Default runtime scans ``./.agents/skills``::

        agent = Agent("a", config=config, plugins=[SkillPlugin()])

    Custom install directory::

        agent = Agent("a", config=config, plugins=[SkillPlugin("./skills")])

    Args:
        runtimes: Zero or more :class:`SkillRuntime` instances or paths to skill
            directories. Empty uses the default :class:`LocalRuntime`.
        middleware: Tool middleware applied to the skill tools.
    """
    toolkit = SkillsToolkit(*runtimes, middleware=middleware)
    skills = toolkit.merged_skills()

    if not skills:
        # No skills: omit the catalog and register no dead tools (spec Step 3).
        return Plugin()

    catalog = "\n".join(_skill_xml(s) for s in skills)
    prompt = (
        "The following skills provide specialized instructions for specific tasks.\n"
        "When a task matches a skill's description, call load_skill(name) to load its\n"
        "full instructions before proceeding.\n"
        f"<available_skills>\n{catalog}\n</available_skills>"
    )

    # Gate the activation tools on capability: only register a tool when at least
    # one skill can actually use it, so the model never sees a dead tool.
    tools = [toolkit.load_skill()]
    if any(s.resources for s in skills):
        tools.append(toolkit.read_skill_resource())
    if any(s.scripts for s in skills):
        tools.append(toolkit.run_skill_script())

    return Plugin(tools=tuple(tools), prompt=prompt)


def _skill_xml(skill: Skill) -> str:
    """Render one catalog entry as a ``<skill>`` XML element.

    Values are XML-escaped so a description containing ``&``/``<``/``>`` cannot
    break the surrounding ``<available_skills>`` block.
    """
    location = escape(skill.location) if skill.location else ""
    return (
        "  <skill>\n"
        f"    <name>{escape(skill.name)}</name>\n"
        f"    <description>{escape(skill.metadata.description)}</description>\n"
        f"    <location>{location}</location>\n"
        "  </skill>"
    )
