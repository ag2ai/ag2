# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from a2a.types import Message, Role, Task, TaskState

from autogen.beta.events.input_events import Input

from .messages import a2a_message_to_inputs, text_from_message

TASK_STATE_TO_FINISH_REASON: dict[int, str] = {
    TaskState.TASK_STATE_COMPLETED: "stop",
    TaskState.TASK_STATE_CANCELED: "cancelled",
    TaskState.TASK_STATE_FAILED: "error",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input_required",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth_required",
}


def user_messages(task: Task) -> list[Message]:
    """Return the user-role messages from a task's history, in order."""
    return [m for m in task.history if m.role == Role.ROLE_USER]


def initial_inputs(task: Task) -> list[Input]:
    """Extract the agent's initial inputs (first user message) from a task."""
    msgs = user_messages(task)
    if not msgs:
        return []
    return a2a_message_to_inputs(msgs[0])


def hitl_replay_queue(task: Task) -> list[Message]:
    """Return follow-up user messages (those after the initial request) in order.

    Each one is replayed against an ``input(...)`` request the agent made on a
    previous attempt. The full ``Message`` (not just text) is returned so the
    executor can inspect ``Message.metadata`` for client-side tool results.
    """
    msgs = user_messages(task)
    return list(msgs[1:])


def hitl_replay_text_queue(task: Task) -> list[str]:
    """Plain-text view of the replay queue (drops metadata)."""
    return [text_from_message(m) for m in hitl_replay_queue(task)]


def finish_reason_for(state: int) -> str:
    """Map an A2A ``TaskState`` enum value to an OpenAI-style ``finish_reason`` string."""
    return TASK_STATE_TO_FINISH_REASON.get(state, TaskState.Name(state))
