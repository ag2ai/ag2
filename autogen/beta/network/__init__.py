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
# Events — re-exported from framework core for backward compatibility
from autogen.beta.events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    TaskProgress,
    TaskRequest,
    TaskResult,
    UnknownEvent,
)

from .convenience import Network

# Events — network-specific
from .events import (
    DelegationError,
    DelegationRejected,
    DelegationRequest,
    DelegationResult,
    SchedulerTriggerFired,
    TopicMessage,
    TopicSubscription,
    TopicUnsubscription,
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
    "ActorInfo",
    "AggregationCompleted",
    "BasePlugin",
    "BufferedChannel",
    "Channel",
    "CompactionCompleted",
    "Conditional",
    "ConflictResolver",
    "DefaultPriority",
    "DefaultPriorityScheme",
    "DelegationError",
    "DelegationRejected",
    "DelegationRequest",
    "DelegationResult",
    "Envelope",
    "EventRegistry",
    "Fanout",
    "FormattedEvent",
    "HighestPriorityWins",
    "HttpChannel",  # noqa: F822
    "Hub",
    "HubContext",
    "LocalChannel",
    "LocalLock",
    "LocalRegistry",
    "Lock",
    "Network",
    "NetworkPolicy",
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
    "SchedulerTriggerFired",
    "TaskProgress",
    "TaskRequest",
    "TaskResult",
    "TelemetryPlugin",
    "TopicInboxPolicy",
    "TopicMessage",
    "TopicOverflow",
    "TopicPlugin",
    "TopicSubscription",
    "TopicUnsubscription",
    "Topology",
    "UnknownEvent",
    "register_event",
)
