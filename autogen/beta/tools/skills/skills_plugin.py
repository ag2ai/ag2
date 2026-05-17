# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.plugin import Plugin

from .local_skills import SkillsToolkit
from .runtime import LocalRuntime, SkillRuntime


class SkillsPlugin(Plugin):
    """Plugin that pre-loads local skills into an agent.

    Pre-loads the named skills' ``SKILL.md`` instructions into the agent's
    system prompt and adds :class:`SkillsToolkit` tools so the agent can
    execute skill scripts.

    Unlike using :class:`SkillsToolkit` alone — which requires the agent to
    call ``load_skill()`` at runtime — ``SkillsPlugin`` makes the skill
    instructions immediately available without any extra tool calls.

    Args:
        *skills:    Names of skills to pre-load.  Each must be installed in
                    the runtime's skill directory.  Pass no names to skip
                    pre-loading (toolkit tools are still added).
        runtime:    A :class:`SkillRuntime`, a directory path, or ``None``
                    for the default ``.agents/skills`` directory.
        middleware: Optional :class:`ToolMiddleware` instances forwarded to
                    :class:`SkillsToolkit`.

    Example::

        agent = Agent(
            "assistant",
            plugins=[SkillsPlugin("web-search", "code-review")],
        )
    """

    def __init__(
        self,
        *skills: str,
        runtime: SkillRuntime | str | os.PathLike[str] | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _runtime: SkillRuntime = LocalRuntime.ensure_runtime(runtime) if runtime is not None else LocalRuntime()

        skill_sections: list[str] = []
        for name in skills:
            content = _runtime.load(name)
            skill_sections.append(f"## Skill: {name}\n\n{content}")

        prompt = "\n\n".join(skill_sections)
        toolkit = SkillsToolkit(runtime=_runtime, middleware=middleware)

        super().__init__(prompt=prompt, tools=[toolkit])
