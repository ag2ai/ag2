# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from collections.abc import AsyncGenerator, Iterable
from contextlib import asynccontextmanager, suppress
from weakref import WeakValueDictionary

from ag2 import aggregate as aggregation
from ag2 import compact as compaction
from ag2.annotations import Context
from ag2.events import BaseEvent, EventLogFailed
from ag2.middleware.base import BaseMiddleware, MiddlewareFactory
from ag2.plugin import Plugin
from ag2.tools.final import tool
from ag2.utils import AGENT_CONTEXT_DEPENDENCY_KEY

from ._maintenance import _AggregationMiddlewareFactory, _CompactionMiddlewareFactory, _effective_store
from .base import KnowledgeStore
from .bootstrap import DefaultBootstrap, StoreBootstrap
from .config import KnowledgeConfig
from .log import EventLogWriter


class _KnowledgeTool:
    def __init__(self, store: KnowledgeStore | None = None) -> None:
        self.store = store

    async def execute(self, action: str, ctx: Context, path: str = "/", content: str = "") -> str:
        """Manage your knowledge store.

        Actions:
            read   - Read file at path.
            write  - Write content to path.
            list   - List entries at path.
            delete - Delete file at path.
        """
        store = _effective_store(self.store, ctx)
        if action == "read":
            result = await store.read(path)
            return result if result is not None else f"Not found: {path}"
        elif action == "write":
            if not content:
                return "Error: content is required for write action."
            await store.write(path, content)
            return f"Written to {path}"
        elif action == "list":
            entries = await store.list(path)
            if not entries:
                return f"Empty: {path}"
            skill = await store.read(f"{path.rstrip('/')}/SKILL.md")
            listing = "\n".join(entries)
            if skill:
                listing = f"{skill}\n---\n{listing}"
            return listing
        elif action == "delete":
            await store.delete(path)
            return f"Deleted: {path}"
        return f"Unknown action: {action}. Available: read, write, list, delete."


class _KnowledgeMiddlewareFactory:
    def __init__(self, config: KnowledgeConfig, legacy_store: KnowledgeStore | None = None) -> None:
        self.config = config
        self._legacy_store = legacy_store
        # Active callers own the locks; custom stores need no weakref/hash support.
        self._locks: WeakValueDictionary[int, asyncio.Lock] = WeakValueDictionary()
        self._configured_store_initialized = False

    def __call__(self, event: BaseEvent, context: Context) -> BaseMiddleware:
        return BaseMiddleware(event, context)

    @asynccontextmanager
    async def scope(self, event: BaseEvent, context: Context) -> AsyncGenerator[None]:
        store = _effective_store(self._legacy_store, context)
        actor = context.dependencies[AGENT_CONTEXT_DEPENDENCY_KEY].name
        configured_store = self.config.store
        uses_configured_store = store is configured_store
        if not uses_configured_store or not self._configured_store_initialized:
            lock = self._locks.get(id(store))
            if lock is None:
                lock = asyncio.Lock()
                self._locks[id(store)] = lock
            async with lock:
                if not uses_configured_store or not self._configured_store_initialized:
                    if not await store.exists("/.initialized"):
                        await store.write("/.initialized", actor)
                        bootstrap = self.config.bootstrap or DefaultBootstrap(mention_tool=self.config.expose_tool)
                        await bootstrap.bootstrap(store, actor)
                    # Only the plugin's own store is memoized. Overrides use their
                    # sentinel on subsequent calls without being kept alive here.
                    if uses_configured_store:
                        self._configured_store_initialized = True
        yield
        if self.config.write_event_log:
            try:
                await EventLogWriter(store).persist(context.stream.id, await context.stream.history.get_events())
            except Exception as exc:
                logging.getLogger(__name__).exception("Event log persistence failed for %s", actor)
                with suppress(Exception):
                    await context.send(EventLogFailed(agent=actor, error_type=type(exc).__name__, error=str(exc)))


class KnowledgePlugin(Plugin):
    """Persistent knowledge, optional model tool, and automatic maintenance.

    ``store`` is a dependency default. Explicit Agent/invocation dependencies
    override it consistently for tools, policies, bootstrap, and maintenance.
    Construction performs no store I/O; bootstrap runs on execution scope entry.

    Args:
        store: Backing knowledge store, contributed as a dependency default.
        expose_tool: Expose the model's read/write/list/delete tool (default True).
        write_event_log: Persist the final stream log (default True).
        bootstrap: Custom store initialization, or the default layout when None.
        compact: Optional history compaction strategy.
        compact_trigger: Firing rules for the compaction strategy; required with ``compact``.
        aggregate: Optional memory aggregation strategy; required by an active ``aggregate_trigger``.
        aggregate_trigger: Firing rules for the aggregation strategy.

    Raises:
        ValueError: If a strategy and its trigger are not configured together.

    At most one KnowledgePlugin may apply to an agent invocation (constructor
    and invocation plugins combined). A delegated worker's own KnowledgePlugin
    takes precedence over the store inherited from its parent.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        expose_tool: bool = True,
        write_event_log: bool = True,
        bootstrap: StoreBootstrap | None = None,
        compact: "compaction.CompactStrategy | None" = None,
        compact_trigger: "compaction.CompactTrigger | None" = None,
        aggregate: "aggregation.AggregateStrategy | None" = None,
        aggregate_trigger: "aggregation.AggregateTrigger | None" = None,
    ) -> None:
        config = KnowledgeConfig(
            store=store,
            expose_tool=expose_tool,
            write_event_log=write_event_log,
            bootstrap=bootstrap,
            compact=compact,
            compact_trigger=compact_trigger,
            aggregate=aggregate,
            aggregate_trigger=aggregate_trigger,
        )
        super().__init__(
            dependencies={KnowledgeStore: store},
            tools=[tool(_KnowledgeTool().execute, name="knowledge")] if expose_tool else [],
            middleware=_knowledge_middleware(config),
        )


def _knowledge_middleware(
    config: KnowledgeConfig, legacy_store: KnowledgeStore | None = None
) -> list[MiddlewareFactory]:
    factories: list[MiddlewareFactory] = [_KnowledgeMiddlewareFactory(config, legacy_store)]
    if config.compact is not None:
        if config.compact_trigger is None:
            raise ValueError("Knowledge compaction requires compact_trigger alongside compact.")
        factories.append(_CompactionMiddlewareFactory(config.compact, config.compact_trigger, legacy_store))
    if (trigger := config.aggregate_trigger) and (
        trigger.every_n_turns > 0 or trigger.every_n_events > 0 or trigger.on_end
    ):
        if config.aggregate is None:
            raise ValueError("Knowledge aggregate_trigger requires an aggregate strategy.")
        factories.append(_AggregationMiddlewareFactory(config.aggregate, trigger, legacy_store))
    return factories


def validate_knowledge_middleware(factories: Iterable[MiddlewareFactory]) -> None:
    """Reject more than one knowledge configuration in one execution scope.

    Every KnowledgePlugin contributes a store default and a ``knowledge`` tool;
    a second one would silently lose both to the first, so it is an error.
    """
    knowledge = [f for f in factories if isinstance(f, _KnowledgeMiddlewareFactory)]
    if len(knowledge) < 2:
        return
    if any(f._legacy_store is not None for f in knowledge):
        raise ValueError("Agent(knowledge=...) cannot be combined with KnowledgePlugin; migrate to plugins only.")
    raise ValueError(
        "Only one KnowledgePlugin may apply to an agent invocation; combine stores in one KnowledgeStore instead."
    )
