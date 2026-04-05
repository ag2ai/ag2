# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for BaseEvent.to_dict() / from_dict() round-trip serialization."""

import pytest

from autogen.beta.compact import CompactionSummary
from autogen.beta.events import ModelRequest
from autogen.beta.events.alert import ObserverAlert, Severity
from autogen.beta.network.events import (
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


@pytest.mark.parametrize(
    "event",
    [
        ModelRequest(content="hello"),
        DelegationRequest(source="a", target="b", task="do it"),
        DelegationResult(source="a", target="b", result="done"),
        DelegationRejected(source="a", target="b", task="t", reason="rejected"),
        DelegationError(source="a", target="b", task="t", error="boom"),
        ObserverAlert(source="obs", severity=Severity.WARNING, message="warn"),
        ObserverStarted(name="obs1"),
        ObserverCompleted(name="obs1"),
        TaskRequest(task="research", task_name="task-1"),
        TaskProgress(task_name="task-1", content="working..."),
        TaskResult(task="research", task_name="task-1", result="done", usage={"tokens": 100}),
        SchedulerTriggerFired(watch_id="w1", target="mon", task="check"),
        TopicMessage(topic="news", sender="writer", message="update", data={"key": "val"}),
        TopicSubscription(actor="reader", topic="news"),
        TopicUnsubscription(actor="reader", topic="news"),
        CompactionSummary(summary="Earlier context...", event_count=50),
        CompactionCompleted(actor="agent", strategy="TailWindowCompact", events_before=100, events_after=50),
        AggregationCompleted(actor="agent", strategy="ConversationSummaryAggregate", event_count=75, llm_calls=1),
        UnknownEvent(type_name="some.module.Foo", data={"x": 1}),
    ],
    ids=lambda e: type(e).__name__,
)
def test_event_round_trip(event) -> None:
    """Every network event type should survive to_dict/from_dict round-trip."""
    data = event.to_dict()
    assert isinstance(data, dict)

    reconstructed = type(event).from_dict(data)
    assert reconstructed.to_dict() == data


def test_nested_event_round_trip() -> None:
    """Events containing nested BaseEvent fields should round-trip correctly."""
    from autogen.beta.network.policies.network import FormattedEvent

    original = ModelRequest(content="hello")
    fe = FormattedEvent(content="formatted", original=original)
    data = fe.to_dict()
    reconstructed = FormattedEvent.from_dict(data)
    assert reconstructed.content == "formatted"
    assert reconstructed.original is not None
    assert reconstructed.original.content == "hello"
