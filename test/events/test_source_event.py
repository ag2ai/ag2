# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.events import BaseEvent, ChildEvent, DownEvent, Usage, UsageEvent
from ag2.usage import UsageReport


def test_local_event_is_its_own_source() -> None:
    event = UsageEvent(Usage(total_tokens=30))

    assert event.source_event is event


@pytest.mark.parametrize("depth", [1, 2])
def test_descendant_source_survives_observation_and_replay(depth: int) -> None:
    source = UsageEvent(Usage(total_tokens=30), model="worker-model")
    observation: BaseEvent = source
    for _ in range(depth):
        observation = ChildEvent(
            inner=observation,
            task_id="child",
            path=("child",),
            agent_name="worker",
            origin_event_id=source.event_id,
        )

    assert observation.source_event is source
    payload = observation.to_dict()
    replayed = ChildEvent.from_dict(payload)
    assert replayed.source_event == source
    assert "source_event" not in payload


def test_intervention_audit_is_its_own_source_without_counting_its_payload() -> None:
    audit = DownEvent(
        inner=UsageEvent(Usage(total_tokens=30)),
        task_id="child",
        path=("child",),
        agent_name="worker",
    )

    assert audit.source_event is audit
    assert UsageReport.from_events([audit]).total == Usage()
