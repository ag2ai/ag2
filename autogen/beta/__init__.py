# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from fast_depends import Depends

from .actor import Actor, KnowledgeConfig, TaskConfig
from .agent import Agent, AgentReply
from .aggregate import (
    AggregateStrategy,
    AggregateTrigger,
    ConversationSummaryAggregate,
    WorkingMemoryAggregate,
)
from .annotations import Context, Inject, Variable
from .assembly import AssemblerMiddleware, AssemblyPolicy
from .compact import (
    CompactStrategy,
    CompactTrigger,
    CompactionSummary,
    SummarizeCompact,
    TailWindowCompact,
)
from .knowledge import (
    DefaultBootstrap,
    EventLogWriter,
    KnowledgeStore,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
    StoreBootstrap,
)
from .events import BaseEvent
from .events.alert import HaltEvent, ObserverAlert, Severity
from .events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    TaskProgress,
    TaskRequest,
    TaskResult,
    UnknownEvent,
)
from .observer import BaseObserver, Observer
from .observers import LoopDetector, TokenMonitor
from .policies import (
    AlertPolicy,
    ConversationPolicy,
    EpisodicMemoryPolicy,
    SlidingWindowPolicy,
    TokenBudgetPolicy,
    WorkingMemoryPolicy,
)
from .response import PromptedSchema, ResponseSchema, response_schema
from .scheduler import Scheduler, WatchStatus
from .state import MemoryStateStore, StateStore
from .stream import MemoryStream
from .tools import ToolResult, tool
from .watch import (
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

__all__ = (
    # Core agent
    "Agent",
    "AgentReply",
    "BaseEvent",
    "Context",
    "Depends",
    "Inject",
    "MemoryStream",
    "PromptedSchema",
    "ResponseSchema",
    "ToolResult",
    "Variable",
    "response_schema",
    "tool",
    # Actor (promoted from network)
    "Actor",
    "KnowledgeConfig",
    "TaskConfig",
    # Assembly
    "AssemblerMiddleware",
    "AssemblyPolicy",
    # Alert system
    "ObserverAlert",
    "Severity",
    "HaltEvent",
    "AlertPolicy",
    # Lifecycle events (promoted from network)
    "AggregationCompleted",
    "CompactionCompleted",
    "ObserverCompleted",
    "ObserverStarted",
    "TaskProgress",
    "TaskRequest",
    "TaskResult",
    "UnknownEvent",
    # Policies
    "ConversationPolicy",
    "EpisodicMemoryPolicy",
    "SlidingWindowPolicy",
    "TokenBudgetPolicy",
    "WorkingMemoryPolicy",
    # Observer
    "BaseObserver",
    "Observer",
    "LoopDetector",
    "TokenMonitor",
    # Knowledge
    "KnowledgeStore",
    "MemoryKnowledgeStore",
    "LockedKnowledgeStore",
    "EventLogWriter",
    "StoreBootstrap",
    "DefaultBootstrap",
    # State
    "StateStore",
    "MemoryStateStore",
    # Compact
    "CompactStrategy",
    "CompactTrigger",
    "CompactionSummary",
    "TailWindowCompact",
    "SummarizeCompact",
    # Aggregate
    "AggregateStrategy",
    "AggregateTrigger",
    "ConversationSummaryAggregate",
    "WorkingMemoryAggregate",
    # Watch
    "Watch",
    "EventWatch",
    "BatchWatch",
    "IntervalWatch",
    "DelayWatch",
    "CronWatch",
    "WindowWatch",
    "AllOf",
    "AnyOf",
    "Sequence",
    # Scheduler
    "Scheduler",
    "WatchStatus",
)
