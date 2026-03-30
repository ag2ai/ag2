# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Network Framework — autonomous agent network infrastructure.

Public API for the network framework. Import everything from here::

    from autogen.beta.network import Actor, Hub, Signal, EventWatch

All primitives (infrastructure protocols, priority schemes, channels)
are exported from this package. For submodule access::

    from autogen.beta.network.primitives.infra import StateStore
"""

# Layer 3: Building Blocks
from .actor import Actor
from .assembler import AssemblerMiddleware, AssemblyPolicy

# Channels
from .convenience import Network

# Layer 2: Events
from .events import (
    AggregationCompleted,
    CompactionCompleted,
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
    TopicMessage,
    TopicSubscription,
    TopicUnsubscription,
    UnknownEvent,
)
from .hub import Hub, RegistrationHandle
from .observer import BaseObserver, Observer

# Built-in observers and plugins
from .observers import LoopDetector, TokenMonitor
from .plugins import RateLimiter, TelemetryPlugin

# Assembly policies
from .policies import (
    ConversationPolicy,
    EpisodicMemoryPolicy,
    NetworkPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
    TopicInboxPolicy,
    TopicOverflow,
    WorkingMemoryPolicy,
)
from .primitives.aggregate import (
    AggregateStrategy,
    AggregateTrigger,
    ConversationSummaryAggregate,
    WorkingMemoryAggregate,
)
from .primitives.channel import BufferedChannel, Channel, LocalChannel, PriorityChannel
from .primitives.compact import (
    CompactStrategy,
    CompactTrigger,
    CompactionSummary,
    SummarizeCompact,
    TailWindowCompact,
)

# Layer 2: Primitives — Envelope & Channel
from .primitives.envelope import Envelope, EventRegistry, register_event

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

# Layer 2: Primitives — Knowledge
from .primitives.knowledge import (
    DefaultBootstrap,
    EventLogWriter,
    KnowledgeStore,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
    StoreBootstrap,
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

# Layer 2: Primitives — Watch
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
    "AggregateStrategy",
    "AggregateTrigger",
    "AggregationCompleted",
    "AllOf",
    "AnyOf",
    "AssemblerMiddleware",
    "AssemblyPolicy",
    "BaseObserver",
    "BasePlugin",
    "BatchWatch",
    "BufferedChannel",
    "Cache",
    "CallHandler",
    "Channel",
    "CompactStrategy",
    "CompactTrigger",
    "CompactionCompleted",
    "CompactionSummary",
    "Conditional",
    "ConflictResolver",
    "ConversationPolicy",
    "ConversationSummaryAggregate",
    "CronWatch",
    "DefaultBootstrap",
    "DefaultPriority",
    "DefaultPriorityScheme",
    "DelayWatch",
    "DelegationError",
    "DelegationRejected",
    "DelegationRequest",
    "DelegationResult",
    "EmitToStream",
    "Envelope",
    "EpisodicMemoryPolicy",
    "EventLogWriter",
    "EventRegistry",
    "EventWatch",
    "Fanout",
    "HaltEvent",
    "HaltOnFatal",
    "HighestPriorityWins",
    "HttpChannel",  # noqa: F822
    "Hub",
    "HubContext",
    "InjectToPrompt",
    "IntervalWatch",
    "KnowledgeStore",
    "LocalChannel",
    "LocalLock",
    "LocalRegistry",
    "Lock",
    "LockedKnowledgeStore",
    "LoopDetector",
    "MemoryCache",
    "MemoryKnowledgeStore",
    "MemoryStateStore",
    "Network",
    "NetworkPolicy",
    "Observer",
    "ObserverCompleted",
    "ObserverStarted",
    "Pipeline",
    "Plugin",
    "PriorityChannel",
    "PriorityScheme",
    "ProcessResult",
    "RateLimiter",
    "RegistrationHandle",
    "Registry",
    "RemoteAgent",
    "RemoteAgentReply",
    "RouteDecision",
    "Scheduler",
    "SchedulerTriggerFired",
    "Sequence",
    "Severity",
    "Signal",
    "SignalPolicy",
    "SlidingWindowPolicy",
    "StateStore",
    "StoreBootstrap",
    "SummarizeCompact",
    "TailWindowCompact",
    "TaskProgress",
    "TaskRequest",
    "TaskResult",
    "TelemetryPlugin",
    "TokenBudgetPolicy",
    "TokenMonitor",
    "TopicInboxPolicy",
    "TopicMessage",
    "TopicOverflow",
    "TopicSubscription",
    "TopicUnsubscription",
    "Topology",
    "UnknownEvent",
    "Watch",
    "WatchStatus",
    "WindowWatch",
    "WorkingMemoryAggregate",
    "WorkingMemoryPolicy",
    "register_event",
)
