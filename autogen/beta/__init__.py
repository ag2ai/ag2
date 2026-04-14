# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from fast_depends import Depends

from .actor import Actor, AgentReply, KnowledgeConfig, TaskConfig
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
from .events import (
    AudioInput,
    BaseEvent,
    BinaryInput,
    DocumentInput,
    ImageInput,
    TextInput,
    VideoInput,
)
from .events.alert import HaltEvent, ObserverAlert, Severity
from .events.lifecycle import (
    AggregationCompleted,
    CompactionCompleted,
    ObserverCompleted,
    ObserverStarted,
    UnknownEvent,
)
from .events.task_events import (
    TaskCompleted,
    TaskFailed,
    TaskProgress,
    TaskStarted,
)
from .knowledge import (
    ChangeCallback,
    ChangeSubscription,
    DefaultBootstrap,
    DiskKnowledgeStore,
    EventLogWriter,
    KnowledgeStore,
    LockedKnowledgeStore,
    MemoryKnowledgeStore,
    NoopChangeSubscription,
    RedisKnowledgeStore,
    SqliteKnowledgeStore,
    StoreBootstrap,
)
from .observer import BaseObserver, Observer, observer
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
    "Actor",
    "AgentReply",
    "AggregateStrategy",
    "AggregateTrigger",
    "AggregationCompleted",
    "AlertPolicy",
    "AllOf",
    "AnyOf",
    "AssemblerMiddleware",
    "AssemblyPolicy",
    "AudioInput",
    "BaseEvent",
    "BaseObserver",
    "BatchWatch",
    "BinaryInput",
    "ChangeCallback",
    "ChangeSubscription",
    "CompactStrategy",
    "CompactTrigger",
    "CompactionCompleted",
    "CompactionSummary",
    "Context",
    "ConversationPolicy",
    "ConversationSummaryAggregate",
    "CronWatch",
    "DefaultBootstrap",
    "DelayWatch",
    "Depends",
    "DiskKnowledgeStore",
    "DocumentInput",
    "EpisodicMemoryPolicy",
    "EventLogWriter",
    "EventWatch",
    "HaltEvent",
    "ImageInput",
    "Inject",
    "IntervalWatch",
    "KnowledgeConfig",
    "KnowledgeStore",
    "LockedKnowledgeStore",
    "LoopDetector",
    "MemoryKnowledgeStore",
    "MemoryStateStore",
    "MemoryStream",
    "NoopChangeSubscription",
    "Observer",
    "ObserverAlert",
    "ObserverCompleted",
    "ObserverStarted",
    "PromptedSchema",
    "RedisKnowledgeStore",
    "ResponseSchema",
    "Scheduler",
    "Sequence",
    "Severity",
    "SlidingWindowPolicy",
    "SqliteKnowledgeStore",
    "StateStore",
    "StoreBootstrap",
    "SummarizeCompact",
    "TailWindowCompact",
    "TaskCompleted",
    "TaskConfig",
    "TaskFailed",
    "TaskProgress",
    "TaskStarted",
    "TextInput",
    "TokenBudgetPolicy",
    "TokenMonitor",
    "ToolResult",
    "UnknownEvent",
    "Variable",
    "VideoInput",
    "Watch",
    "WatchStatus",
    "WindowWatch",
    "WorkingMemoryAggregate",
    "WorkingMemoryPolicy",
    "observer",
    "response_schema",
    "tool",
)
