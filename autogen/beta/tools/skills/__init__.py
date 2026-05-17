# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from .local_skills import SkillsToolkit
from .runtime import LocalRuntime
from .skill_search import SkillSearchToolkit, SkillsClientConfig

__all__ = (
    "LocalRuntime",
    "SkillSearchToolkit",
    "SkillsClientConfig",
    "SkillsPlugin",  # noqa: F822 — lazy-loaded via __getattr__ below
    "SkillsToolkit",
)


def __getattr__(name: str) -> object:
    if name == "SkillsPlugin":
        from .skills_plugin import SkillsPlugin

        return SkillsPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
