# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Hub — discovery, routing, and plugin pipeline for a network of actors.

The Hub is the network center. It provides:
- Registry: actors register with capabilities for discovery
- Router: all inter-actor delegation flows through the Hub
- Stream: the Hub's own event stream for cross-actor observation
- Headless mode: pure routing without an initiating LLM call
- Remote serving: expose agents over HTTP for cross-server delegation
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from aiohttp import web

from autogen.beta.agent import Agent, AgentReply
from autogen.beta.context import Context, Stream
from autogen.beta.events.types import ModelMessageChunk, ModelResponse
from autogen.beta.state import MemoryStateStore, StateStore
from autogen.beta.stream import MemoryStream
from autogen.beta.tools.final import tool
from autogen.beta.tools.tool import Tool

from .events import (
    DelegationCancelled,
    DelegationError,
    DelegationProgress,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    DelegationStarted,
)
from .primitives.channel import Channel, LocalChannel, PriorityChannel
from .primitives.envelope import Envelope
from .primitives.infra import ActorInfo, LocalRegistry, Registry
from .primitives.priority import ConflictResolver, PriorityScheme
from .topology import HubContext, Plugin, Topology, _normalize

logger = logging.getLogger(__name__)

_delegation_depth: contextvars.ContextVar[int] = contextvars.ContextVar("delegation_depth", default=0)

_delegation_source: contextvars.ContextVar[str] = contextvars.ContextVar("delegation_source", default="")

_delegation_metadata: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("delegation_metadata", default={})

# Delegation ID of the current background execution, read by the network
# ``report_progress`` tool so agents don't need to pass it explicitly.
_current_delegation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_delegation_id", default=""
)

# StateStore key prefix for background delegation state.
_DELEGATION_KEY_PREFIX = "delegation:"


def _delegation_key(delegation_id: str) -> str:
    return f"{_DELEGATION_KEY_PREFIX}{delegation_id}"


def _format_delegation_state(state: dict[str, Any]) -> str:
    """Render a persisted delegation state as a human-readable block."""
    lines = [
        f"delegation_id: {state.get('delegation_id', '?')}",
        f"state:         {state.get('state', '?')}",
        f"target:        {state.get('target', '?')}",
        f"source:        {state.get('source', '?')}",
    ]
    task = state.get("task")
    if task:
        # Keep the task compact so list_pending stays scannable
        trimmed = task if len(task) <= 240 else f"{task[:237]}..."
        lines.append(f"task:          {trimmed}")
    progress = state.get("progress")
    progress_msg = state.get("progress_message") or ""
    if progress is not None or progress_msg:
        pct = f"{progress:.0%}" if isinstance(progress, (int, float)) else "?"
        suffix = f" — {progress_msg}" if progress_msg else ""
        lines.append(f"progress:      {pct}{suffix}")
    result = state.get("result")
    if result:
        trimmed_r = result if len(result) <= 500 else f"{result[:497]}..."
        lines.append(f"result:        {trimmed_r}")
    error = state.get("error")
    if error:
        lines.append(f"error:         {error}")
    return "\n".join(lines)


@dataclass
class RegistrationHandle:
    """Handle returned by Hub.register(). Provides convenience unregister."""

    name: str
    _hub: Hub

    async def unregister(self) -> None:
        """Unregister this agent from the Hub."""
        await self._hub.unregister(self.name)


class Hub:
    """Discovery, routing, and plugin pipeline for a network of actors.

    Supports two modes:
    - **Agent mode**: hub.ask(actor, task) — invokes an actor with network tools
    - **Headless mode**: hub.delegate(source, target, task) — pure routing

    Example::

        hub = Hub()
        hub.register(researcher, capabilities=["research", "analysis"])
        hub.register(writer, capabilities=["writing", "editing"])

        reply = await hub.ask(researcher, "Research and write a report")

    When ``propagate_metadata=True``, metadata passed to ``hub.ask(metadata=...)``
    is set on the delegation envelope and bridged into agent variables at each
    delegation boundary. This lets tools receive cross-agent state via ``Variable``
    annotation — invisible to the LLM, automatically propagated through the
    delegation chain via ``Envelope.child()``.

    Example::

        hub = Hub(propagate_metadata=True)
        reply = await hub.ask(coordinator, task, metadata={"project_id": "abc"})
        # Every delegated agent's tools can now access project_id via:
        #   project_id: Annotated[str, Variable("project_id")]
    """

    def __init__(
        self,
        *,
        stream: Stream | None = None,
        topology: Topology | None = None,
        plugins: Iterable[Plugin] = (),
        channel: Channel | None = None,
        registry: Registry | None = None,
        state_store: StateStore | None = None,
        priority_scheme: PriorityScheme | None = None,
        conflict_resolver: ConflictResolver | None = None,
        max_delegation_depth: int = 5,
        propagate_metadata: bool = False,
    ) -> None:
        self._propagate_metadata = propagate_metadata
        self._registry = registry or LocalRegistry()
        # Auto-wire PriorityChannel when a priority_scheme is provided but
        # no explicit channel was given.  This ensures that envelopes with
        # priority metadata are actually delivered in priority order.
        if channel is not None:
            self._channel = channel
        elif priority_scheme is not None:
            self._channel = PriorityChannel(scheme=priority_scheme)
        else:
            self._channel = LocalChannel()
        self._state_store = state_store or MemoryStateStore()
        self._priority_scheme = priority_scheme
        self._conflict_resolver = conflict_resolver
        self._stream = stream or MemoryStream()
        self._context = Context(self._stream)
        self._agents: dict[str, Agent] = {}
        self._max_depth = max_delegation_depth
        self._topology = topology
        self._plugins: list[Plugin] = list(plugins)
        self._hub_context = HubContext(self)
        self._additional_tasks: set[asyncio.Task[None]] = set()
        # Background delegations (request_background) keyed by delegation_id.
        self._background_tasks: dict[str, asyncio.Task[None]] = {}

        # Cross-actor knowledge exposure
        self._exposed_paths: dict[str, list[str]] = {}  # actor -> exposed path prefixes

        # Server state (populated by serve())
        self._server_runner: web.AppRunner | None = None
        self._server_site: web.TCPSite | None = None

        # Install topology plugins and system plugins
        if self._topology:
            self._topology.install_plugins(self)
        for plugin in self._plugins:
            plugin.install(self)

    async def _emit(self, event: Any) -> None:
        """Emit an event on the Hub stream, logging but not raising subscriber errors."""
        try:
            await self._context.send(event)
        except Exception:
            logger.exception("Hub stream subscriber error while emitting %s", type(event).__name__)

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    async def register(
        self,
        agent: Agent,
        capabilities: list[str] | None = None,
        description: str = "",
        exposed_paths: list[str] | None = None,
    ) -> RegistrationHandle:
        """Register an agent with the network. Returns a handle for unregistering."""
        if agent.name in self._agents:
            logger.warning(
                "Agent '%s' is already registered — overwriting. "
                "Call hub.unregister('%s') first to avoid this warning.",
                agent.name,
                agent.name,
            )
        info = ActorInfo(
            name=agent.name,
            capabilities=capabilities or [],
            description=description,
        )
        await self._registry.register(agent.name, info)
        self._agents[agent.name] = agent
        if exposed_paths:
            self._exposed_paths[agent.name] = list(exposed_paths)
        return RegistrationHandle(name=agent.name, _hub=self)

    async def unregister(self, name: str) -> None:
        """Remove an agent from the network."""
        await self._registry.unregister(name)
        self._agents.pop(name, None)
        self._exposed_paths.pop(name, None)
        # Clean up topic subscriptions via TopicPlugin
        tp = self._topic_plugin
        if tp is not None:
            tp.cleanup_actor(name)

    async def discover(self, capability: str = "") -> list[ActorInfo]:
        """Find registered agents, optionally filtered by capability."""
        return await self._registry.discover(capability)

    # ------------------------------------------------------------------
    # Network tools (injected into agents)
    # ------------------------------------------------------------------

    def _build_network_tools(self, caller: str = "") -> list[Tool]:
        """Build the consolidated network tool for a specific caller."""
        hub = self

        @tool
        async def network(
            action: str,
            target: str = "",
            topic: str = "",
            message: str = "",
        ) -> str:
            """Communicate over the agent network.

            Actions:
                discover           - Find agents. target=capability filter (optional).
                request            - Delegate task (blocking). target=agent, message=task.
                request_background - Spawn a non-blocking delegation. target=agent,
                                     message=task. Returns a delegation_id.
                check_status       - Query a background delegation. message=delegation_id.
                cancel             - Cancel a background delegation. message=delegation_id.
                list_pending       - List in-flight background delegations started by
                                     this caller. target="all" to include terminal
                                     states, otherwise only pending/running.
                report_progress    - Update progress from inside a background delegation.
                                     message=human-readable status. topic (optional) =
                                     numeric fraction 0.0-1.0. Uses the current
                                     delegation contextvar — no id required.
                publish            - Publish to topic. topic=topic name, message=content.
                subscribe          - Subscribe to a topic. topic=topic name.
                topics             - List all active topics.
                query              - Read from another actor's knowledge. target=agent, message=path.
                query_list         - List another actor's knowledge entries. target=agent, message=path.
            """
            if action == "discover":
                agents = await hub.discover(target)
                infos = [a for a in agents if a.name != caller]
                if not infos:
                    return "No other agents found."
                lines = []
                for a in infos:
                    caps = ", ".join(a.capabilities) if a.capabilities else "general"
                    desc = f" - {a.description}" if a.description else ""
                    lines.append(f"- {a.name} [{caps}]{desc}")
                return "\n".join(lines)

            elif action == "request":
                if not target:
                    return "Error: target is required for request action."
                if target == caller:
                    return "Error: cannot delegate to yourself."
                if not message:
                    return "Error: message is required for request action."
                # Optionally propagate metadata through the delegation chain via contextvar
                metadata = None
                if hub._propagate_metadata:
                    current = _delegation_metadata.get()
                    if current:
                        metadata = dict(current)
                return await hub._delegate(target, message, source=caller, metadata=metadata)

            elif action == "request_background":
                if not target:
                    return "Error: target is required for request_background action."
                if target == caller:
                    return "Error: cannot delegate to yourself."
                if not message:
                    return "Error: message is required for request_background action."
                metadata = None
                if hub._propagate_metadata:
                    current = _delegation_metadata.get()
                    if current:
                        metadata = dict(current)
                try:
                    delegation_id = await hub.request_background(
                        target, message, source=caller, metadata=metadata
                    )
                except ValueError as exc:
                    return f"Error: {exc}"
                return (
                    f"Spawned background delegation to '{target}'. "
                    f"delegation_id: {delegation_id}"
                )

            elif action == "check_status":
                if not message:
                    return "Error: message (delegation_id) is required for check_status action."
                state = await hub.check_delegation(message)
                if state is None:
                    return f"Error: delegation '{message}' not found."
                return _format_delegation_state(state)

            elif action == "cancel":
                if not message:
                    return "Error: message (delegation_id) is required for cancel action."
                ok = await hub.cancel_delegation(message)
                if ok:
                    return f"Cancelled delegation '{message}'."
                # Might have already completed — still useful to report the final state
                state = await hub.check_delegation(message)
                if state is None:
                    return f"Error: delegation '{message}' not found."
                return (
                    f"Delegation '{message}' already finished in state "
                    f"'{state.get('state', '?')}'."
                )

            elif action == "list_pending":
                include_terminal = target == "all"
                entries = await hub.list_pending_delegations(
                    source=caller, include_terminal=include_terminal
                )
                if not entries:
                    return "No pending delegations."
                return "\n\n".join(_format_delegation_state(e) for e in entries)

            elif action == "report_progress":
                delegation_id = _current_delegation_id.get()
                if not delegation_id:
                    return (
                        "Error: report_progress can only be called from within a "
                        "background delegation (no current delegation_id)."
                    )
                progress_value: float | None = None
                if topic:
                    try:
                        progress_value = float(topic)
                    except ValueError:
                        return f"Error: progress value '{topic}' is not a valid float."
                ok = await hub.report_progress(
                    delegation_id, progress=progress_value, message=message
                )
                if not ok:
                    return f"Error: delegation '{delegation_id}' not found."
                return f"Reported progress for delegation '{delegation_id}'."

            elif action == "publish":
                tp = hub._topic_plugin
                if tp is None:
                    return "Topics not configured. Install TopicPlugin on the Hub."
                if not topic:
                    return "Error: topic is required for publish action."
                if not message:
                    return "Error: message is required for publish action."
                await tp.publish(caller, topic, message)
                return f"Published to topic '{topic}'."

            elif action == "subscribe":
                tp = hub._topic_plugin
                if tp is None:
                    return "Topics not configured. Install TopicPlugin on the Hub."
                if not topic:
                    return "Error: topic is required for subscribe action."
                await tp.subscribe_topic(caller, topic)
                return f"Subscribed to topic '{topic}'."

            elif action == "topics":
                tp = hub._topic_plugin
                if tp is None:
                    return "Topics not configured. Install TopicPlugin on the Hub."
                topic_list = await tp.list_topics()
                if not topic_list:
                    return "No active topics."
                return "\n".join(f"- {t}" for t in topic_list)

            elif action == "query":
                if not target or not message:
                    return "Error: target (agent name) and message (path) required for query."
                content = await hub.query_knowledge(caller, target, message)
                if content is None:
                    return f"Not accessible: {target}:{message} (not found or not exposed)."
                return content

            elif action == "query_list":
                if not target:
                    return "Error: target (agent name) required for query_list."
                path = message or "/"
                entries = await hub.list_knowledge(caller, target, path)
                if entries is None:
                    return f"Not accessible: {target}:{path} (not found or not exposed)."
                if not entries:
                    return f"Empty: {target}:{path}"
                return "\n".join(entries)

            else:
                return f"Unknown action: {action}. Available: discover, request, publish, subscribe, topics, query, query_list."

        return [network]

    # ------------------------------------------------------------------
    # Delegation
    # ------------------------------------------------------------------

    async def _delegate(
        self,
        to_agent: str,
        task: str,
        *,
        source: str = "",
        priority: Any = None,
        metadata: dict[str, Any] | None = None,
        delegation_id: str = "",
        **kwargs: Any,
    ) -> str:
        """Internal: route a task to a registered agent.

        If ``delegation_id`` is provided, it is stamped on all emitted
        delegation events so subscribers (and the StateStore-backed
        background registry) can correlate the events with a specific job.
        """
        agent = self._agents.get(to_agent)
        if not agent:
            available = ", ".join(self._agents.keys()) or "(none)"
            return f"Error: agent '{to_agent}' not found. Available: {available}"

        depth = _delegation_depth.get()
        if depth >= self._max_depth:
            await self._emit(
                DelegationRejected(
                    source=source,
                    target=to_agent,
                    task=task,
                    reason=f"Maximum delegation depth ({self._max_depth}) reached.",
                    delegation_id=delegation_id,
                )
            )
            return (
                f"Error: maximum delegation depth ({self._max_depth}) reached. "
                "Cannot delegate further to prevent infinite loops."
            )

        # Create envelope for the delegation
        envelope = Envelope(
            event=DelegationRequest(
                source=source, target=to_agent, task=task, delegation_id=delegation_id
            ),
            sender=source,
            recipient=to_agent,
            priority=priority,
            metadata=metadata or {},
        )

        # Run through topology pipeline if configured
        additional: list[Envelope] = []
        if self._topology:
            topology_result = await self._topology.process(envelope, self._hub_context)

            # Normalize: split into primary + additional envelopes
            primary, additional = _normalize(topology_result)

            if primary is None:
                # Primary rejected — still dispatch additional (reject-with-side-effects)
                self._dispatch_additional(additional)
                await self._emit(
                    DelegationRejected(
                        source=source,
                        target=to_agent,
                        task=task,
                        reason="Rejected by topology pipeline.",
                        delegation_id=delegation_id,
                    )
                )
                return "Error: delegation rejected by Hub topology."

            # Topology may have rerouted
            to_agent = primary.recipient or to_agent
            rerouted_agent = self._agents.get(to_agent)
            if rerouted_agent is None:
                available = ", ".join(self._agents.keys()) or "(none)"
                await self._emit(
                    DelegationRejected(
                        source=source,
                        target=to_agent,
                        task=task,
                        reason=f"Rerouted agent '{to_agent}' not found.",
                        delegation_id=delegation_id,
                    )
                )
                return f"Error: rerouted agent '{to_agent}' not found. Available: {available}"
            agent = rerouted_agent
            envelope = primary

            # Ensure envelope event target is consistent with routed recipient.
            # Topology plugins typically modify envelope.recipient but not the
            # inner event — fix up so channel subscribers see correct target.
            # Preserve any task modifications made by topology plugins.
            if isinstance(envelope.event, DelegationRequest) and envelope.event.target != to_agent:
                envelope.event = DelegationRequest(
                    source=source,
                    target=to_agent,
                    task=envelope.event.task,
                    delegation_id=delegation_id,
                )

        # Dispatch additional delegations from topology (fire-and-forget, parallel)
        if additional:
            self._dispatch_additional(additional)

        # Use the (possibly modified) task from the envelope event
        effective_task = envelope.event.task if isinstance(envelope.event, DelegationRequest) else task

        # Emit on Hub stream and send through Channel
        await self._emit(
            DelegationRequest(
                source=source, target=to_agent, task=effective_task, delegation_id=delegation_id
            )
        )

        depth_token = _delegation_depth.set(depth + 1)
        source_token = _delegation_source.set(source)
        meta_token = _delegation_metadata.set(envelope.metadata)
        try:
            await self._channel.send(envelope)
            network_tools = self._build_network_tools(caller=to_agent)
            # Bridge envelope metadata → agent variables for tool injection (opt-in)
            if self._propagate_metadata and envelope.metadata:
                existing_vars = kwargs.pop("variables", None) or {}
                kwargs["variables"] = {**envelope.metadata, **existing_vars}
            reply = await agent.ask(effective_task, tools=network_tools, **kwargs)
            result = reply.body or ""
            result_event = DelegationResult(
                source=source, target=to_agent, result=result, delegation_id=delegation_id
            )
            await self._emit(result_event)
            await self._channel.send(envelope.child(event=result_event, sender=to_agent, recipient=source))
            return result
        except Exception as e:
            error_msg = f"Error during delegation to '{to_agent}': {e}"
            error_event = DelegationError(
                source=source,
                target=to_agent,
                task=effective_task,
                error=str(e),
                delegation_id=delegation_id,
            )
            await self._emit(error_event)
            await self._channel.send(envelope.child(event=error_event, sender=to_agent, recipient=source))
            return error_msg
        finally:
            _delegation_depth.reset(depth_token)
            _delegation_source.reset(source_token)
            _delegation_metadata.reset(meta_token)

    def _dispatch_additional(self, envelopes: list[Envelope]) -> None:
        """Dispatch additional delegations from topology as background tasks.

        Each additional envelope is executed as an independent delegation
        through the full Hub path (depth tracking, topology, events).
        Errors are logged and emitted on the hub stream but do not affect
        the primary delegation. Tasks are tracked so close() can await them.
        """
        for env in envelopes:
            target = env.recipient
            task_text = getattr(env.event, "task", "")
            source = env.sender or ""
            if target and task_text:
                t = asyncio.create_task(
                    self._execute_additional(target, task_text, source),
                    name=f"additional-delegation-{target}",
                )
                self._additional_tasks.add(t)
                t.add_done_callback(self._additional_tasks.discard)

    async def _execute_additional(self, target: str, task: str, source: str) -> None:
        """Execute a single additional delegation with error handling."""
        try:
            await self._delegate(target, task, source=source)
        except Exception:
            logger.exception("Additional delegation to '%s' failed", target)

    async def delegate(
        self,
        source: str,
        target: str,
        task: str,
        *,
        priority: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Headless mode: route a task directly without an initiating LLM call.

        Use for infrastructure Hubs that only route traffic, run plugins,
        and manage registry — zero LLM cost for the Hub itself.
        """
        return await self._delegate(target, task, source=source, priority=priority, metadata=metadata)

    # ------------------------------------------------------------------
    # Background delegation — StateStore-backed, non-blocking, durable
    # ------------------------------------------------------------------

    async def request_background(
        self,
        to_agent: str,
        task: str,
        *,
        source: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Spawn a non-blocking delegation. Returns the delegation_id immediately.

        The delegation runs as a background task on the current event loop.
        Its state is persisted to the Hub's StateStore under the key
        ``delegation:{id}`` so it survives out-of-process consumers crashing.
        On Hub shutdown, all in-flight background tasks are cancelled and
        marked ``cancelled`` in the store.

        Query the outcome with :meth:`check_delegation` or cancel via
        :meth:`cancel_delegation`. Subscribers to the Hub stream will
        receive ``DelegationStarted`` on spawn and the usual
        ``DelegationResult`` / ``DelegationError`` on completion.
        """
        if not to_agent:
            raise ValueError("to_agent is required for request_background")
        if not task:
            raise ValueError("task is required for request_background")

        delegation_id = uuid4().hex
        now = time.time()
        initial_state: dict[str, Any] = {
            "delegation_id": delegation_id,
            "state": "pending",
            "source": source,
            "target": to_agent,
            "task": task,
            "metadata": dict(metadata or {}),
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
            "progress": None,
            "progress_message": "",
        }
        await self._state_store.set(_delegation_key(delegation_id), initial_state)
        await self._emit(
            DelegationStarted(
                source=source,
                target=to_agent,
                task=task,
                delegation_id=delegation_id,
            )
        )

        task_handle = asyncio.create_task(
            self._run_background_delegation(delegation_id, to_agent, task, source, metadata),
            name=f"delegation-{delegation_id}",
        )
        self._background_tasks[delegation_id] = task_handle
        task_handle.add_done_callback(lambda t: self._background_tasks.pop(delegation_id, None))

        return delegation_id

    async def _run_background_delegation(
        self,
        delegation_id: str,
        to_agent: str,
        task: str,
        source: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Execute a background delegation and update StateStore with the outcome."""
        # Isolate the delegation chain from whatever context spawned us — this
        # task is not nested under the caller's depth, it starts a fresh chain.
        depth_token = _delegation_depth.set(0)
        source_token = _delegation_source.set(source or "")
        meta_token = _delegation_metadata.set(dict(metadata or {}))
        delegation_token = _current_delegation_id.set(delegation_id)
        try:
            await self._update_delegation_state(delegation_id, state="running")
            result = await self._delegate(
                to_agent,
                task,
                source=source,
                metadata=metadata,
                delegation_id=delegation_id,
            )
            # ``_delegate`` returns synthetic error strings instead of raising;
            # detect those so ``state`` correctly reflects the outcome.
            if isinstance(result, str) and result.startswith("Error"):
                await self._update_delegation_state(
                    delegation_id, state="failed", error=result
                )
            else:
                await self._update_delegation_state(
                    delegation_id, state="completed", result=result
                )
        except asyncio.CancelledError:
            await self._update_delegation_state(
                delegation_id,
                state="cancelled",
                error="Background delegation was cancelled",
            )
            await self._emit(
                DelegationCancelled(
                    source=source,
                    target=to_agent,
                    delegation_id=delegation_id,
                    reason="cancelled",
                )
            )
            raise
        except Exception as exc:
            logger.exception("Background delegation %s failed unexpectedly", delegation_id)
            await self._update_delegation_state(
                delegation_id, state="failed", error=str(exc)
            )
        finally:
            _delegation_depth.reset(depth_token)
            _delegation_source.reset(source_token)
            _delegation_metadata.reset(meta_token)
            _current_delegation_id.reset(delegation_token)

    async def _update_delegation_state(
        self,
        delegation_id: str,
        *,
        state: str | None = None,
        result: str | None = None,
        error: str | None = None,
        progress: float | None = None,
        progress_message: str | None = None,
    ) -> dict[str, Any] | None:
        """Merge new fields into the persisted delegation state."""
        key = _delegation_key(delegation_id)
        current = await self._state_store.get(key)
        if current is None:
            return None
        if state is not None:
            current["state"] = state
        if result is not None:
            current["result"] = result
        if error is not None:
            current["error"] = error
        if progress is not None:
            current["progress"] = progress
        if progress_message is not None:
            current["progress_message"] = progress_message
        current["updated_at"] = time.time()
        await self._state_store.set(key, current)
        return current

    async def check_delegation(self, delegation_id: str) -> dict[str, Any] | None:
        """Return the current state of a background delegation, or None if unknown."""
        return await self._state_store.get(_delegation_key(delegation_id))

    async def cancel_delegation(self, delegation_id: str) -> bool:
        """Cancel an in-flight background delegation.

        Returns ``True`` if the task was cancelled, ``False`` if the delegation
        is unknown or has already finished.
        """
        task_handle = self._background_tasks.get(delegation_id)
        if task_handle is None:
            return False
        task_handle.cancel()
        try:
            await task_handle
        except (asyncio.CancelledError, Exception):
            pass
        return True

    async def list_pending_delegations(
        self,
        *,
        source: str | None = None,
        include_terminal: bool = False,
    ) -> list[dict[str, Any]]:
        """List background delegations, optionally filtered by ``source`` agent.

        By default only non-terminal entries (``pending``, ``running``) are
        returned. Pass ``include_terminal=True`` to also include
        ``completed`` / ``failed`` / ``cancelled`` / ``orphaned`` entries.

        Requires the StateStore to implement ``scan``. Stores that do not
        support enumeration return an empty list.
        """
        scan = getattr(self._state_store, "scan", None)
        if scan is None:
            return []
        try:
            keys = await scan(_DELEGATION_KEY_PREFIX)
        except NotImplementedError:
            return []
        terminal = {"completed", "failed", "cancelled", "orphaned"}
        results: list[dict[str, Any]] = []
        for key in keys:
            state = await self._state_store.get(key)
            if not isinstance(state, dict):
                continue
            if source is not None and state.get("source") != source:
                continue
            if not include_terminal and state.get("state") in terminal:
                continue
            results.append(state)
        # Stable order: oldest first
        results.sort(key=lambda s: s.get("created_at", 0.0))
        return results

    async def recover_orphaned_delegations(self) -> list[str]:
        """Mark any pending/running delegation state in the StateStore as orphaned.

        Call this once on Hub startup after wiring a persistent StateStore.
        Entries from a previous process that were mid-flight will be flagged
        so callers can reason about them rather than assuming they completed.

        Returns the list of delegation_ids that were recovered.
        """
        scan = getattr(self._state_store, "scan", None)
        if scan is None:
            return []
        try:
            keys = await scan(_DELEGATION_KEY_PREFIX)
        except NotImplementedError:
            return []
        orphaned: list[str] = []
        for key in keys:
            state = await self._state_store.get(key)
            if not isinstance(state, dict):
                continue
            if state.get("state") not in ("pending", "running"):
                continue
            delegation_id = state.get("delegation_id", "")
            state["state"] = "orphaned"
            state["error"] = "Hub restarted while delegation was in flight"
            state["updated_at"] = time.time()
            await self._state_store.set(key, state)
            await self._emit(
                DelegationError(
                    source=state.get("source", ""),
                    target=state.get("target", ""),
                    task=state.get("task", ""),
                    error="orphaned on Hub restart",
                    delegation_id=delegation_id,
                )
            )
            orphaned.append(delegation_id)
        return orphaned

    async def report_progress(
        self,
        delegation_id: str,
        *,
        progress: float | None = None,
        message: str = "",
    ) -> bool:
        """Record a progress update for a background delegation.

        Delegatees (or adapters wrapping them) call this to announce incremental
        progress. Emits ``DelegationProgress`` on the Hub stream and merges the
        fields into the StateStore entry so observers and later ``check_delegation``
        readers see the update.

        Returns ``False`` if the delegation is unknown.
        """
        updated = await self._update_delegation_state(
            delegation_id,
            progress=progress,
            progress_message=message,
        )
        if updated is None:
            return False
        await self._emit(
            DelegationProgress(
                source=updated.get("source", ""),
                target=updated.get("target", ""),
                delegation_id=delegation_id,
                progress=progress,
                message=message,
            )
        )
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ask(
        self,
        agent: Agent | str,
        message: str,
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentReply:
        """Start a task through the Hub.

        Injects network tools (discover_agents, delegate_to) so the agent
        can find and communicate with other registered agents.

        Args:
            agent: The agent to ask, by name or reference.
            message: The task or question.
            metadata: Key-value pairs to propagate through the delegation chain.
                Flows via envelope metadata → agent variables, so tools can
                receive values via ``Variable`` annotation without exposing
                them to the LLM tool schema.
            **kwargs: Passed through to ``agent.ask()``.
        """
        resolved: Agent
        if isinstance(agent, str):
            resolved_agent = self._agents.get(agent)
            if not resolved_agent:
                raise KeyError(f"Agent '{agent}' not registered with this Hub.")
            resolved = resolved_agent
        else:
            resolved = agent

        network_tools = self._build_network_tools(caller=resolved.name)
        existing_tools = list(kwargs.pop("tools", []))
        # Inject metadata as variables so the initiating agent's tools can access them (opt-in)
        if self._propagate_metadata and metadata:
            existing_vars = kwargs.pop("variables", None) or {}
            kwargs["variables"] = {**metadata, **existing_vars}
        # Set metadata contextvar so network tool can propagate it through delegations
        meta_token = _delegation_metadata.set(metadata) if self._propagate_metadata and metadata else None
        try:
            return await resolved.ask(message, tools=existing_tools + network_tools, **kwargs)
        finally:
            if meta_token is not None:
                _delegation_metadata.reset(meta_token)

    async def ask_stream(
        self,
        agent: Agent | str,
        message: str,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream text chunks as they are generated by the agent.

        Yields ``ModelMessageChunk.content`` strings in real-time, enabling
        low-latency consumption (e.g. feeding a TTS pipeline before the full
        response is ready).

        The underlying ``ask()`` call runs to completion — tool calls, policies,
        aggregation — all work identically to the non-streaming path.

        Example::

            async for chunk in hub.ask_stream(twin, "What happened today?"):
                tts.feed(chunk)
        """
        resolved: Agent
        if isinstance(agent, str):
            resolved_agent = self._agents.get(agent)
            if not resolved_agent:
                raise KeyError(f"Agent '{agent}' not registered with this Hub.")
            resolved = resolved_agent
        else:
            resolved = agent

        # Create a per-call stream so subscribers don't leak across turns
        stream = kwargs.pop("stream", None) or MemoryStream()
        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def _on_chunk(event: ModelMessageChunk) -> None:
            await chunk_queue.put(event.content)

        # Track final responses that are NOT tool_use (the actual answer)
        _got_final = False

        async def _on_response(event: ModelResponse) -> None:
            nonlocal _got_final
            if event.finish_reason != "tool_use":
                _got_final = True

        # Subscribe before starting the ask to avoid missing early chunks
        with (
            stream.where(ModelMessageChunk).sub_scope(_on_chunk),
            stream.where(ModelResponse).sub_scope(_on_response),
        ):
            network_tools = self._build_network_tools(caller=resolved.name)
            existing_tools = list(kwargs.pop("tools", []))

            # Run ask() in a background task so we can yield chunks as they arrive
            ask_task = asyncio.create_task(
                resolved.ask(
                    message,
                    stream=stream,
                    tools=existing_tools + network_tools,
                    **kwargs,
                )
            )

            try:
                while True:
                    # Race between: next chunk arriving, or ask completing
                    get_chunk = asyncio.ensure_future(chunk_queue.get())
                    done, _ = await asyncio.wait(
                        [get_chunk, ask_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if get_chunk in done:
                        value = get_chunk.result()
                        if value is not None:
                            yield value
                    else:
                        get_chunk.cancel()

                    if ask_task in done:
                        # Drain any remaining chunks queued before task completed
                        while not chunk_queue.empty():
                            value = chunk_queue.get_nowait()
                            if value is not None:
                                yield value
                        # Propagate any exception from the ask
                        ask_task.result()
                        break
            except BaseException:
                ask_task.cancel()
                raise

    async def close(self) -> None:
        """Clean up plugins, channel, server, and remote agent sessions.

        Called automatically by ``serve()`` context manager, or call manually
        when using the Hub without ``serve()``.
        """
        # Cancel and await in-flight background delegations. Each cancelled
        # task updates the StateStore to state='cancelled' via its own
        # CancelledError handler in _run_background_delegation.
        for t in list(self._background_tasks.values()):
            t.cancel()
        if self._background_tasks:
            await asyncio.gather(
                *self._background_tasks.values(), return_exceptions=True
            )
        self._background_tasks.clear()

        # Cancel and await in-flight additional delegations
        for t in self._additional_tasks:
            t.cancel()
        if self._additional_tasks:
            await asyncio.gather(*self._additional_tasks, return_exceptions=True)
        self._additional_tasks.clear()

        # Close remote agent sessions
        from .remote import RemoteAgent

        for agent in self._agents.values():
            if isinstance(agent, RemoteAgent):
                await agent.close()

        for plugin in self._plugins:
            plugin.uninstall()
        if self._topology:
            self._topology.uninstall_plugins()
        await self._channel.close()

        # Close the stream if it supports it (e.g. RedisStream holds connections)
        if hasattr(self._stream, "close"):
            await self._stream.close()

        # Stop HTTP server if running
        if self._server_site:
            await self._server_site.stop()
            self._server_site = None
        if self._server_runner:
            await self._server_runner.cleanup()
            self._server_runner = None

    # ------------------------------------------------------------------
    # HTTP server for remote delegation
    # ------------------------------------------------------------------

    async def _handle_delegate_request(self, request: web.Request) -> web.Response:
        """Handle incoming delegation request from a RemoteAgent."""
        from aiohttp import web

        try:
            data = await request.json()
            agent_name = data.get("agent", "")
            task = data.get("task", "")
            source = data.get("source", "remote")
            metadata = data.get("metadata")

            if not agent_name or not task:
                return web.json_response(
                    {"status": "error", "reason": "Missing 'agent' or 'task' field"},
                    status=400,
                )

            if agent_name not in self._agents:
                available = ", ".join(self._agents.keys()) or "(none)"
                return web.json_response(
                    {
                        "status": "error",
                        "reason": f"Agent '{agent_name}' not found. Available: {available}",
                    },
                    status=404,
                )

            # Route through the full delegation pipeline so remote delegations
            # get event emission, topology processing, and depth tracking.
            result = await self._delegate(agent_name, task, source=source, metadata=metadata)

            return web.json_response({"status": "ok", "result": result})

        except Exception as e:
            logger.exception("Failed to handle delegation request")
            return web.json_response(
                {"status": "error", "reason": str(e)},
                status=500,
            )

    async def _handle_discover_request(self, request: web.Request) -> web.Response:
        """Return registered local agents for remote discovery."""
        from aiohttp import web

        from .remote import RemoteAgent

        capability = request.query.get("capability", "")
        agents = await self.discover(capability)

        # Only expose local agents — don't re-advertise remote proxies
        result = []
        for info in agents:
            agent = self._agents.get(info.name)
            if isinstance(agent, RemoteAgent):
                continue
            result.append({
                "name": info.name,
                "capabilities": info.capabilities,
                "description": info.description,
            })

        return web.json_response({"agents": result})

    async def _handle_health_request(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        from aiohttp import web

        return web.json_response({
            "status": "healthy",
            "agents": len(self._agents),
        })

    @asynccontextmanager
    async def serve(self, *, host: str | None = None, port: int = 8900):
        """Run the Hub, optionally as an HTTP server for remote delegation.

        When ``host`` is provided, starts an HTTP server that exposes:
        - ``POST /delegate`` — accept delegation requests from RemoteAgents
        - ``GET /discover`` — return registered local agents
        - ``GET /health`` — liveness check

        When ``host`` is None (default), behaves as before — just manages
        lifecycle without starting a server.

        .. note::

            No authentication is applied to HTTP endpoints. Intended for
            trusted networks or development. Production deployments should
            add authentication middleware to the aiohttp app or use a
            reverse proxy.

        Example::

            # Headless (no HTTP server)
            async with hub.serve():
                await asyncio.Event().wait()

            # With HTTP server for remote delegation
            async with hub.serve(host="0.0.0.0", port=8900):
                await asyncio.Event().wait()
        """
        if host is not None:
            from aiohttp import web

            app = web.Application()
            app.router.add_post("/delegate", self._handle_delegate_request)
            app.router.add_get("/discover", self._handle_discover_request)
            app.router.add_get("/health", self._handle_health_request)

            self._server_runner = web.AppRunner(app)
            await self._server_runner.setup()
            self._server_site = web.TCPSite(self._server_runner, host, port)
            await self._server_site.start()
            logger.info("Hub server started on %s:%d", host, port)

        try:
            yield self
        finally:
            await self.close()

    # ------------------------------------------------------------------
    # Remote Hub connection
    # ------------------------------------------------------------------

    async def connect(
        self,
        endpoint: str,
        *,
        timeout: float = 30.0,
        remote_timeout: float = 120.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
    ) -> list[str]:
        """Discover and register agents from a remote Hub.

        Fetches the remote Hub's ``/discover`` endpoint, creates a
        ``RemoteAgent`` proxy for each remote agent, and registers
        them with this Hub.

        Args:
            endpoint: Base URL of the remote Hub (e.g. ``"http://server-a:8900"``).
            timeout: Timeout for the discovery request.
            remote_timeout: Timeout for individual delegation requests to remote agents.
            max_retries: Max retries for remote agent delegation calls.
            retry_delay: Base delay between retries.

        Returns:
            List of agent names that were discovered and registered.

        Example::

            await hub.connect("http://server-a:8900")
            # Now hub can delegate to agents on server-a
        """
        from aiohttp import ClientSession, ClientTimeout

        from .remote import RemoteAgent

        url = f"{endpoint.rstrip('/')}/discover"

        async with ClientSession(timeout=ClientTimeout(total=timeout)) as session, session.get(url) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Failed to discover agents at {url}: HTTP {resp.status}: {body}")
            data = await resp.json()

        registered: list[str] = []
        conflicts: list[str] = []
        for agent_info in data.get("agents", []):
            name = agent_info["name"]
            if name in self._agents:
                conflicts.append(name)
                continue
            capabilities = agent_info.get("capabilities", [])
            description = agent_info.get("description", "")

            remote = RemoteAgent(
                name,
                endpoint,
                capabilities=capabilities,
                description=description,
                timeout=remote_timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
            await self.register(remote, capabilities=capabilities, description=description)
            registered.append(name)

        if conflicts:
            logger.warning(
                "Skipped %d remote agent(s) from %s due to name conflicts "
                "with locally registered agents: %s. Unregister them first "
                "if you want the remote versions.",
                len(conflicts),
                endpoint,
                ", ".join(conflicts),
            )

        logger.info(
            "Connected to %s — registered %d remote agents: %s",
            endpoint,
            len(registered),
            ", ".join(registered),
        )
        return registered

    # ------------------------------------------------------------------
    # Cross-actor knowledge queries
    # ------------------------------------------------------------------

    async def query_knowledge(self, requester: str, target: str, path: str) -> str | None:
        """Read from another actor's knowledge store.

        Returns content if the path is exposed. Returns None if the target
        has no store, the path is not exposed, or the path doesn't exist.
        """
        store = self._get_exposed_store(target, path)
        if store is None:
            return None
        return await store.read(path)

    async def list_knowledge(self, requester: str, target: str, path: str = "/") -> list[str] | None:
        """List entries in another actor's knowledge store.

        Same exposure rules as query_knowledge.
        """
        store = self._get_exposed_store(target, path)
        if store is None:
            return None
        return await store.list(path)

    def _get_exposed_store(self, target: str, path: str) -> Any:
        """Return the target's knowledge store if the path is exposed, else None."""
        agent = self._agents.get(target)
        if agent is None:
            return None

        # Access _knowledge_store (Actor private attr, same package)
        store = getattr(agent, "_knowledge_store", None)
        if store is None:
            return None

        exposed = self._exposed_paths.get(target, [])
        if not any(path.startswith(prefix) for prefix in exposed):
            return None

        return store

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _topic_plugin(self):
        """Find the installed TopicPlugin, if any."""
        from .plugins.topic import TopicPlugin

        for plugin in self._plugins:
            if isinstance(plugin, TopicPlugin):
                return plugin
        return None

    @property
    def stream(self) -> MemoryStream:
        """The Hub's cross-actor event stream."""
        return self._stream

    @property
    def agents(self) -> dict[str, Agent]:
        """Snapshot of currently registered agents."""
        return dict(self._agents)

    @property
    def state_store(self) -> StateStore:
        """The Hub's state store."""
        return self._state_store

    @property
    def priority_scheme(self) -> PriorityScheme | None:
        """The Hub's priority scheme, if configured."""
        return self._priority_scheme

    @property
    def conflict_resolver(self) -> ConflictResolver | None:
        """The Hub's conflict resolver, if configured."""
        return self._conflict_resolver
