# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Network Framework — autonomous agent network infrastructure.

Public API for the network framework. Import everything from here::

    from autogen.beta.network import Actor, Hub, Signal, EventWatch

All primitives (infrastructure protocols, priority schemes, harness, channels)
are exported from this package. For submodule access::

    from autogen.beta.network.primitives.infra import StateStore
"""

# Layer 2: Primitives — Watch
# Layer 3: Building Blocks
from .actor import Actor

# Channels
from .convenience import Network

# Layer 2: Events
from .events import (
    DelegationError,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    ObserverCompleted,
    ObserverStarted,
    SchedulerTriggerFired,
    TaskProgress,
    TaskRequest,
    TaskResult,
)
from .hub import Hub, RegistrationHandle
from .observer import BaseObserver, Observer

# Built-in observers and plugins
from .observers import LoopDetector, TokenMonitor
from .plugins import RateLimiter, TelemetryPlugin
from .primitives.channel import BufferedChannel, Channel, LocalChannel, PriorityChannel

# Layer 2: Primitives — Envelope & Channel
from .primitives.envelope import Envelope, EventRegistry, register_event

# Layer 2: Primitives — Harness
from .primitives.harness import (
    ContextHarness,
    ConversationHarness,
    FormattedEvent,
    HarnessMiddleware,
    NetworkHarness,
)

# Layer 2: Primitives — Infrastructure
from .primitives.infra import (
    ActorInfo,
    Cache,
    LocalLock,
    LocalRegistry,
    Lock,
    MemoryCache,
    MemoryStateStore,
    Registry,
    StateStore,
)

# Layer 2: Primitives — Priority
from .primitives.priority import (
    ConflictResolver,
    DefaultPriority,
    DefaultPriorityScheme,
    HighestPriorityWins,
    PriorityScheme,
)

# Layer 2: Primitives — Signal
from .primitives.signal import (
    CallHandler,
    EmitToStream,
    HaltEvent,
    HaltOnFatal,
    InjectToPrompt,
    Severity,
    Signal,
    SignalPolicy,
)
from .primitives.watch import (
    AllOf,
    AnyOf,
    BatchWatch,
    CronWatch,
    DelayWatch,
    EventWatch,
    IntervalWatch,
    Sequence,
    Watch,
    WindowWatch,
)

# Remote
from .remote import RemoteAgent, RemoteAgentReply
from .scheduler import Scheduler, WatchStatus

# Layer 4: Composition
from .topology import (
    BasePlugin,
    Conditional,
    Fanout,
    HubContext,
    Pipeline,
    Plugin,
    ProcessResult,
    RouteDecision,
    Topology,
)


def __getattr__(name: str) -> object:
    if name == "HttpChannel":
        from .channels.http import HttpChannel

        return HttpChannel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    # Building Blocks
    "Actor",
    "ActorInfo",
    "AllOf",
    "AnyOf",
    "BaseObserver",
    "BasePlugin",
    "BatchWatch",
    "BufferedChannel",
    "Cache",
    "CallHandler",
    "Channel",
    "Conditional",
    "ConflictResolver",
    # Primitives — Harness
    "ContextHarness",
    "ConversationHarness",
    "CronWatch",
    "DefaultPriority",
    "DefaultPriorityScheme",
    "DelayWatch",
    "DelegationError",
    "DelegationRejected",
    "DelegationRequest",
    "DelegationResult",
    "EmitToStream",
    # Primitives — Envelope & Channel
    "Envelope",
    "EventRegistry",
    "EventWatch",
    "Fanout",
    "FormattedEvent",
    "HaltEvent",
    "HaltOnFatal",
    "HarnessMiddleware",
    "HighestPriorityWins",
    "HttpChannel",  # noqa: F822
    "Hub",
    "HubContext",
    "InjectToPrompt",
    "IntervalWatch",
    "LocalChannel",
    "LocalLock",
    "LocalRegistry",
    "Lock",
    "LoopDetector",
    "MemoryCache",
    "MemoryStateStore",
    "Network",
    "NetworkHarness",
    "Observer",
    "ObserverCompleted",
    # Events
    "ObserverStarted",
    "Pipeline",
    # Composition
    "Plugin",
    "PriorityChannel",
    # Primitives — Priority
    "PriorityScheme",
    "ProcessResult",
    "RateLimiter",
    "RegistrationHandle",
    "Registry",
    # Remote
    "RemoteAgent",
    "RemoteAgentReply",
    "RouteDecision",
    "Scheduler",
    "SchedulerTriggerFired",
    "Sequence",
    "Severity",
    # Primitives — Signal
    "Signal",
    "SignalPolicy",
    # Primitives — Infrastructure
    "StateStore",
    "TaskProgress",
    "TaskRequest",
    "TaskResult",
    "TelemetryPlugin",
    # Observers & Plugins
    "TokenMonitor",
    "Topology",
    # Primitives — Watch
    "Watch",
    "WatchStatus",
    "WindowWatch",
    "register_event",
)
