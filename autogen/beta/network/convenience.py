# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Network — convenience class that wires Hub + Scheduler with sensible defaults."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from autogen.beta.agent import Agent, AgentReply
from autogen.beta.context import Context
from autogen.beta.events import BaseEvent
from autogen.beta.stream import MemoryStream

from .hub import Hub, RegistrationHandle
from .primitives.channel import Channel
from .primitives.infra import Registry, StateStore
from .primitives.priority import ConflictResolver, PriorityScheme
from .primitives.watch import Watch
from .scheduler import Scheduler
from .topology import Plugin, Topology


class Network:
    """Convenience: Hub + Scheduler wired together with sensible defaults.

    Example::

        network = Network()
        await network.register(researcher, capabilities=["research"])
        await network.register(writer, capabilities=["writing"])

        network.schedule(IntervalWatch(300), target="researcher", task="Check trends")

        await network.start()
        reply = await network.ask(researcher, "Write a trend report")
        await network.stop()
    """

    def __init__(
        self,
        *,
        stream: MemoryStream | None = None,
        topology: Topology | None = None,
        plugins: Iterable[Plugin] = (),
        channel: Channel | None = None,
        registry: Registry | None = None,
        state_store: StateStore | None = None,
        priority_scheme: PriorityScheme | None = None,
        conflict_resolver: ConflictResolver | None = None,
        max_delegation_depth: int = 5,
    ) -> None:
        self.hub = Hub(
            stream=stream,
            topology=topology,
            plugins=plugins,
            channel=channel,
            registry=registry,
            state_store=state_store,
            priority_scheme=priority_scheme,
            conflict_resolver=conflict_resolver,
            max_delegation_depth=max_delegation_depth,
        )
        self.scheduler = Scheduler(hub=self.hub)

    async def register(
        self,
        agent: Agent,
        capabilities: list[str] | None = None,
        description: str = "",
    ) -> RegistrationHandle:
        """Register an agent with the network. Returns a handle for unregistering."""
        return await self.hub.register(agent, capabilities=capabilities, description=description)

    async def unregister(self, name: str) -> None:
        """Remove an agent from the network."""
        await self.hub.unregister(name)

    def schedule(
        self,
        watch: Watch,
        *,
        target: str = "",
        task: str = "",
        task_factory: Callable[[list[BaseEvent]], str] | None = None,
        callback: Callable[[list[BaseEvent], Context], Awaitable[None]] | None = None,
        priority: Any = None,
    ) -> str:
        """Schedule a watch. Shorthand for scheduler.add()."""
        return self.scheduler.add(
            watch,
            target=target,
            task=task,
            task_factory=task_factory,
            callback=callback,
            priority=priority,
        )

    async def connect(
        self,
        endpoint: str,
        **kwargs: Any,
    ) -> list[str]:
        """Discover and register agents from a remote Hub. Shorthand for hub.connect()."""
        return await self.hub.connect(endpoint, **kwargs)

    async def ask(
        self,
        agent: Agent | str,
        message: str,
        **kwargs: Any,
    ) -> AgentReply:
        """Start a task through the network. Shorthand for hub.ask()."""
        return await self.hub.ask(agent, message, **kwargs)

    async def start(self) -> None:
        """Start the scheduler."""
        await self.scheduler.start()

    async def stop(self) -> None:
        """Stop the scheduler."""
        await self.scheduler.stop()

    async def __aenter__(self) -> Network:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.stop()
        await self.hub.close()
