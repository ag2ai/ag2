# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from .base import BaseEvent, Field


class TaskStarted(BaseEvent):
    task_id: str
    agent_name: str
    objective: str


class TaskCompleted(BaseEvent):
    task_id: str
    agent_name: str
    objective: str
    result: str
    task_stream: Any = Field(default=None)  # Stream reference for inspection


class TaskFailed(BaseEvent):
    task_id: str
    agent_name: str
    objective: str
    error: str
