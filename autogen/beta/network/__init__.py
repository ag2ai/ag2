# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""AG2 Network Module — agent-to-agent communication infrastructure.

Network-specific imports (Hub, delegation, topology, channels, remote).
Framework-core features (Actor, policies, observers, knowledge, etc.)
are now in ``autogen.beta`` directly.

Usage::

    from autogen.beta import Actor, ConversationPolicy, TokenMonitor
    from autogen.beta.network import Hub, Network, Pipeline, RateLimiter
"""

# Convenience
from .convenience import Network

# Events (network-specific only)
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

# Hub
from .hub import Hub, RegistrationHandle

# Plugins
from .plugins import RateLimiter, TelemetryPlugin, TopicPlugin

# Network-specific policies
from .policies import NetworkPolicy, TopicInboxPolicy, TopicOverflow
from .policies.network import FormattedEvent

# Channels & Envelopes
from .primitives.channel import BufferedChannel, Channel, LocalChannel, PriorityChannel
from .primitives.envelope import Envelope, EventRegistry, register_event

# Infrastructure (network-specific)
from .primitives.infra import ActorInfo, LocalLock, LocalRegistry, Lock, Registry

# Priority (deferred, internal)
from .primitives.priority import (
    ConflictResolver,
    DefaultPriority,
    DefaultPriorityScheme,
    HighestPriorityWins,
    PriorityScheme,
)

# Remote
from .remote import RemoteAgent, RemoteAgentReply

# Topology
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
    # Hub & Network
    "Hub",
    "RegistrationHandle",
    "Network",
    # Events
    "AggregationCompleted",
    "CompactionCompleted",
    "DelegationError",
    "DelegationRejected",
    "DelegationRequest",
    "DelegationResult",
    "FormattedEvent",
    "ObserverCompleted",
    "ObserverStarted",
    "SchedulerTriggerFired",
    "TaskProgress",
    "TaskRequest",
    "TaskResult",
    "TopicMessage",
    "TopicSubscription",
    "TopicUnsubscription",
    "UnknownEvent",
    # Channels & Envelopes
    "BufferedChannel",
    "Channel",
    "Envelope",
    "EventRegistry",
    "HttpChannel",  # noqa: F822
    "LocalChannel",
    "PriorityChannel",
    "register_event",
    # Infrastructure
    "ActorInfo",
    "LocalLock",
    "LocalRegistry",
    "Lock",
    "Registry",
    # Priority (internal)
    "ConflictResolver",
    "DefaultPriority",
    "DefaultPriorityScheme",
    "HighestPriorityWins",
    "PriorityScheme",
    # Plugins
    "RateLimiter",
    "TelemetryPlugin",
    "TopicPlugin",
    # Policies (network-specific)
    "NetworkPolicy",
    "TopicInboxPolicy",
    "TopicOverflow",
    # Remote
    "RemoteAgent",
    "RemoteAgentReply",
    # Topology
    "BasePlugin",
    "Conditional",
    "Fanout",
    "HubContext",
    "Pipeline",
    "Plugin",
    "ProcessResult",
    "RouteDecision",
    "Topology",
)
