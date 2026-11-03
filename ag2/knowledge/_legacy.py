# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Temporary Agent(knowledge=...) adapter; remove in 1.4.0."""

from typing import Any

from ag2.plugin import Plugin
from ag2.tools.final import tool

from .base import KnowledgeStore
from .config import KnowledgeConfig
from .plugin import _KnowledgeTool, _knowledge_middleware


class LegacyKnowledgeAdapter:
    def __init__(self, config: KnowledgeConfig) -> None:
        # Legacy knowledge overrides constructor dependencies, unlike plugin defaults.
        self.dependencies: dict[Any, Any] = {KnowledgeStore: config.store}
        self.plugin = Plugin(
            dependencies=self.dependencies,
            middleware=_knowledge_middleware(config, legacy_store=config.store),
        )
        # Preserve hidden-tool exclusion from the deprecated subtask inheritance.
        self.tools = [tool(_KnowledgeTool(config.store).execute, name="knowledge")] if config.expose_tool else []
