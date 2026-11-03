# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Temporary Agent(knowledge=...) adapter; remove in 1.4.0."""

from collections.abc import Iterable

from ag2.middleware.base import MiddlewareFactory
from ag2.plugin import Plugin
from ag2.tools.final import tool

from .base import KnowledgeStore
from .config import KnowledgeConfig
from .plugin import _KnowledgeMiddlewareFactory, _KnowledgeTool, _knowledge_middleware


class LegacyKnowledgeAdapter:
    def __init__(self, config: KnowledgeConfig) -> None:
        self.plugin = Plugin(
            dependencies={KnowledgeStore: config.store},
            middleware=_knowledge_middleware(config, legacy_store=config.store),
        )
        # Preserve hidden-tool exclusion from the deprecated subtask inheritance.
        self.tools = [tool(_KnowledgeTool(config.store).execute, name="knowledge")] if config.expose_tool else []

    def validate(self, factories: Iterable[MiddlewareFactory]) -> None:
        if any(isinstance(f, _KnowledgeMiddlewareFactory) and f.legacy_store is None for f in factories):
            raise ValueError("Agent(knowledge=...) cannot be combined with KnowledgePlugin; migrate to plugins only.")
