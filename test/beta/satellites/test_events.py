# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.satellites.events import (
    SatelliteCompleted,
    SatelliteFlag,
    SatelliteStarted,
    Severity,
    TaskSatelliteRequest,
    TaskSatelliteResult,
)


def test_severity_values():
    assert Severity.INFO == "info"
    assert Severity.WARNING == "warning"
    assert Severity.CRITICAL == "critical"
    assert Severity.FATAL == "fatal"


def test_satellite_flag_creation():
    flag = SatelliteFlag(
        source="test-monitor",
        severity=Severity.WARNING,
        message="Token budget at 80%",
    )
    assert flag.source == "test-monitor"
    assert flag.severity == Severity.WARNING
    assert flag.message == "Token budget at 80%"


def test_satellite_lifecycle_events():
    started = SatelliteStarted(name="token-monitor")
    assert started.name == "token-monitor"

    completed = SatelliteCompleted(name="token-monitor")
    assert completed.name == "token-monitor"


def test_task_satellite_events():
    req = TaskSatelliteRequest(task="research AI", satellite_name="task-abc123")
    assert req.task == "research AI"
    assert req.satellite_name == "task-abc123"

    result = TaskSatelliteResult(
        task="research AI",
        satellite_name="task-abc123",
        result="AI is transforming...",
        usage={"total_tokens": 500},
    )
    assert result.result == "AI is transforming..."
    assert result.usage == {"total_tokens": 500}


def test_task_satellite_result_default_usage():
    result = TaskSatelliteResult(
        task="test",
        satellite_name="sat-1",
        result="done",
    )
    assert result.usage == {}
