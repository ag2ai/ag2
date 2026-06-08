# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from typing import TYPE_CHECKING
from xml.sax.saxutils import escape

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.skills.local_skills.toolkit import SkillsToolkit
from autogen.beta.tools.skills.runtime import SkillRuntime

if TYPE_CHECKING:
    from autogen.beta.agent import Plugin


def SkillPlugin(  # noqa: N802
    runtime: SkillRuntime | str | os.PathLike[str] | None = None,
    *,
    middleware: Iterable[ToolMiddleware] = (),
) -> "Plugin":
    """Skills-spec ``Plugin`` that auto-loads skill metadata into the prompt.

    Follows the ``agentskills.io`` progressive-disclosure pattern, but instead
    of exposing a ``list_skills`` tool call, it injects the skill catalog as an
    ``<available_skills>`` XML block (name + description + location per skill)
    into the system prompt on agent startup. The model discovers what is
    available without spending a tool round-trip, then:

    1. ``load_skill(name)`` — read the full ``SKILL.md`` on demand.
    2. ``read_skill_resource(name, resource)`` — read a bundled resource file.
    3. ``run_skill_script(name, script, args)`` — execute a skill script.

    The catalog and the activation tools are a **construction-time snapshot**:
    the tools constrain ``name`` to the set of skills present when the plugin is
    built, and the catalog lists exactly that set — the two never drift apart.
    When no skills are found, the plugin contributes nothing (no catalog, no
    tools) so the model is not shown dead tools.

    Default runtime scans ``./.agents/skills`` and ``~/.agents/skills``::

        agent = Agent("a", config=config, plugins=[SkillPlugin()])

    Custom install directory::

        agent = Agent("a", config=config, plugins=[SkillPlugin("./skills")])

    Args:
        runtime: A :class:`SkillRuntime`, or a path to a skills directory.
            ``None`` uses the default :class:`LocalRuntime`.
        middleware: Tool middleware applied to the skill tools.
    """
    # prevent circular import
    # TODO: refactor it after moving some parts to "internal" package
    from autogen.beta.agent import Plugin

    toolkit = SkillsToolkit(runtime, middleware=middleware)
    skills = toolkit.discover_skills()

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

    return Plugin(
        tools=(
            toolkit.load_skill(),
            toolkit.read_skill_resource(),
            toolkit.run_skill_script(),
        ),
        prompt=prompt,
    )


def _skill_xml(skill: dict[str, str]) -> str:
    """Render one catalog entry as a ``<skill>`` XML element.

    Values are XML-escaped so a description containing ``&``/``<``/``>`` cannot
    break the surrounding ``<available_skills>`` block.
    """
    return (
        "  <skill>\n"
        f"    <name>{escape(skill['name'])}</name>\n"
        f"    <description>{escape(skill['description'])}</description>\n"
        f"    <location>{escape(skill['location'])}</location>\n"
        "  </skill>"
    )
