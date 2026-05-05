# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""autogen.beta.network — agent registry, durable messaging, and protocol-driven sessions.

V1 ships in three internal milestones (see ``design/PLAN.md``):

* M1 — Foundation: identity, envelope, rule, auth, transport, hub plumbing
* M2 — Consulting loop: first end-to-end LLM-driven session adapter + tools
* M3 — Multi-party + observability: discussion adapter, expectations, full tool surface

Importing ``autogen.beta.network`` is opt-in — bare ``Agent`` continues
to work standalone with no behavioural change when this package is not
imported.
"""

from .auth import AuthAdapter, AuthRegistry, NoAuth, default_registry
from .client import AgentClient, HubClient, NetworkClient
from .hub import Hub
from .transport import (
    AcceptFrame,
    ErrorFrame,
    EventFrame,
    Frame,
    HelloFrame,
    LinkClient,
    LinkEndpoint,
    LocalLink,
    LocalLinkClient,
    LocalLinkEndpoint,
    NotifyFrame,
    PingFrame,
    PongFrame,
    ReceiptFrame,
    SendFrame,
    SubscribeFrame,
    UnsubscribeFrame,
    WelcomeFrame,
    decode_frame,
    encode_frame,
)
from .envelope import (
    EV_ERROR,
    EV_EXPECTATION_VIOLATED,
    EV_PARTICIPANT_REMOVED,
    EV_PEER_RECONNECTED,
    EV_PEER_UNREACHABLE,
    EV_SESSION_CLOSED,
    EV_SESSION_EXPIRED,
    EV_SESSION_IDLE,
    EV_SESSION_INVITE,
    EV_SESSION_INVITE_ACK,
    EV_SESSION_INVITE_REJECT,
    EV_SESSION_OPENED,
    EV_SESSION_QUORUM_CHANGED,
    EV_TASK_ERROR,
    EV_TASK_EXPIRED,
    EV_TASK_PROGRESS,
    EV_TASK_RESULT,
    EV_TASK_STALLED,
    EV_TASK_STARTED,
    EV_TEXT,
    Envelope,
    Priority,
    visible_to,
)
from .errors import (
    AccessDeniedError,
    AuthError,
    InboxFull,
    NetworkError,
    NotFoundError,
    ProtocolError,
)
from .identity import (
    AgentRuntime,
    AuthBlock,
    CostProfile,
    ObservedStat,
    Passport,
    Resume,
    ResumeExample,
)
from .ids import make_id
from .policies import AGENT_CLIENT_DEP, HUB_DEP, SESSION_DEP, TASK_DEP
from .rule import (
    AccessBlock,
    InboxBlock,
    LimitsBlock,
    RateBlock,
    Rule,
    SessionTypeAccess,
    parse_duration,
)

__all__ = (
    "AGENT_CLIENT_DEP",
    "AcceptFrame",
    "AccessBlock",
    "AccessDeniedError",
    "AgentClient",
    "AgentRuntime",
    "AuthAdapter",
    "AuthBlock",
    "AuthError",
    "AuthRegistry",
    "CostProfile",
    "EV_ERROR",
    "EV_EXPECTATION_VIOLATED",
    "EV_PARTICIPANT_REMOVED",
    "EV_PEER_RECONNECTED",
    "EV_PEER_UNREACHABLE",
    "EV_SESSION_CLOSED",
    "EV_SESSION_EXPIRED",
    "EV_SESSION_IDLE",
    "EV_SESSION_INVITE",
    "EV_SESSION_INVITE_ACK",
    "EV_SESSION_INVITE_REJECT",
    "EV_SESSION_OPENED",
    "EV_SESSION_QUORUM_CHANGED",
    "EV_TASK_ERROR",
    "EV_TASK_EXPIRED",
    "EV_TASK_PROGRESS",
    "EV_TASK_RESULT",
    "EV_TASK_STALLED",
    "EV_TASK_STARTED",
    "EV_TEXT",
    "Envelope",
    "ErrorFrame",
    "EventFrame",
    "Frame",
    "HUB_DEP",
    "HelloFrame",
    "Hub",
    "HubClient",
    "InboxBlock",
    "InboxFull",
    "LimitsBlock",
    "LinkClient",
    "LinkEndpoint",
    "LocalLink",
    "LocalLinkClient",
    "LocalLinkEndpoint",
    "NetworkClient",
    "NetworkError",
    "NoAuth",
    "NotFoundError",
    "NotifyFrame",
    "ObservedStat",
    "Passport",
    "PingFrame",
    "PongFrame",
    "Priority",
    "ProtocolError",
    "RateBlock",
    "ReceiptFrame",
    "Resume",
    "ResumeExample",
    "Rule",
    "SESSION_DEP",
    "SendFrame",
    "SessionTypeAccess",
    "SubscribeFrame",
    "TASK_DEP",
    "UnsubscribeFrame",
    "WelcomeFrame",
    "decode_frame",
    "default_registry",
    "encode_frame",
    "make_id",
    "parse_duration",
    "visible_to",
)
