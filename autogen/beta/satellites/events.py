# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from autogen.beta.events.base import BaseEvent, Field


class Severity(str, Enum):
    """Severity levels for satellite flags."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class SatelliteFlag(BaseEvent):
    """Flag emitted by a satellite to notify the planet agent."""

    source: str
    severity: str
    message: str


class SatelliteStarted(BaseEvent):
    """Emitted when a satellite attaches to the stream."""

    name: str


class SatelliteCompleted(BaseEvent):
    """Emitted when a satellite detaches from the stream."""

    name: str


class TaskSatelliteRequest(BaseEvent):
    """Emitted when the planet spawns a task satellite."""

    task: str
    satellite_name: str


class TaskSatelliteProgress(BaseEvent):
    """Streamed progress from a running task satellite."""

    satellite_name: str
    content: str


class TaskSatelliteResult(BaseEvent):
    """Emitted when a task satellite completes its work."""

    task: str
    satellite_name: str
    result: str
    usage: dict = Field(default_factory=dict)
