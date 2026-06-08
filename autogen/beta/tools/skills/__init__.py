# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any

from .local_skills import SkillsToolkit
from .runtime import LocalRuntime
from .skill_search import SkillSearchToolkit, SkillsClientConfig

if TYPE_CHECKING:
    from .local_skills.plugin import SkillPlugin

__all__ = (
    "LocalRuntime",
    "SkillPlugin",
    "SkillSearchToolkit",
    "SkillsClientConfig",
    "SkillsToolkit",
)


# SkillPlugin builds an autogen.beta.agent.Plugin. agent.py eagerly imports
# this package during its own load (agent -> tools -> skills), so importing
# Plugin at module top would be a circular import. Defer it to attribute
# access, by which point agent.py is fully initialised.
def __getattr__(name: str) -> Any:
    if name == "SkillPlugin":
        from .local_skills.plugin import SkillPlugin

        return SkillPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
