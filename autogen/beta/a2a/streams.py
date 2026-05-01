# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from a2a.client import Client
from a2a.types import (
    GetTaskRequest,
    StreamResponse,
    SubscribeToTaskRequest,
    Task,
    TaskState,
)

from autogen.beta.context import ConversationContext

from .errors import A2AAuthRequiredError, A2AReconnectError
from .mappers import REASONING_ARTIFACT_NAME, message_metadata, task_artifact_update_to_events
from .types import TERMINAL_TASK_STATES, TRANSPORT_ERRORS


@dataclass(slots=True)
class StreamOutcome:
    """Result of consuming one A2A streaming or polling session."""

    text: str = ""
    reasoning: str = ""
    task: Task | None = None
    input_required: bool = False
    input_prompt: str | None = None
    input_metadata: dict[str, Any] = field(default_factory=dict)


async def drain(
    stream: AsyncIterator[StreamResponse],
    context: ConversationContext,
    outcome: StreamOutcome,
) -> BaseException | None:
    """Consume one ``StreamResponse`` iterator into ``outcome``.

    Returns the connection error encountered (if any) so the caller can decide
    whether to reconnect.
    """
    try:
        async for response in stream:
            payload = response.WhichOneof("payload")

            if payload == "message":
                # Out-of-task agent message — informational, no state to track.
                continue

            if payload == "task":
                outcome.task = response.task

            elif payload == "status_update":
                # Apply the status update to the locally tracked task so that
                # ``input_required`` / terminal checks below see fresh state.
                # Synthetic Task built from the update has no `history` —
                # reconnect-to-task-with-client-tools may lose the original
                # client tool stubs (known limitation; full reconnect should
                # `get_task` to recover history if that matters).
                upd = response.status_update
                if outcome.task is None or outcome.task.id != upd.task_id:
                    outcome.task = Task(id=upd.task_id, context_id=upd.context_id, status=upd.status, history=[])
                else:
                    outcome.task.status.CopyFrom(upd.status)

            elif payload == "artifact_update":
                upd = response.artifact_update
                if outcome.task is None:
                    outcome.task = Task(id=upd.task_id, context_id=upd.context_id, history=[])
                is_reasoning = upd.artifact.name == REASONING_ARTIFACT_NAME
                for ev in task_artifact_update_to_events(upd):
                    if is_reasoning:
                        outcome.reasoning += ev.content
                    else:
                        outcome.text += ev.content
                    await context.send(ev)

            if outcome.task is not None:
                state = outcome.task.status.state
                if state == TaskState.TASK_STATE_AUTH_REQUIRED:
                    raise A2AAuthRequiredError(outcome.task)
                if state in TERMINAL_TASK_STATES:
                    return None
                if state == TaskState.TASK_STATE_INPUT_REQUIRED:
                    outcome.input_required = True
                    _read_input_prompt(outcome.task, outcome)
                    return None
        return None
    except TRANSPORT_ERRORS as exc:
        return exc


async def reconnect(
    client: Client,
    *,
    outcome: StreamOutcome,
    context: ConversationContext,
    max_attempts: int,
    backoff: float,
) -> None:
    """Re-establish a session after a transport drop.

    A2A 1.0 collapses the streaming/polling split: ``Client.subscribe`` always
    works for resuming an in-flight task. If subscribe itself fails (server
    cannot stream), fall back to a single ``get_task`` poll.
    """
    last_error: BaseException | None = None
    attempts = 0
    while _needs_reconnect(outcome) and attempts < max_attempts:
        assert outcome.task is not None
        attempts += 1
        try:
            try:
                last_error = await drain(
                    client.subscribe(SubscribeToTaskRequest(id=outcome.task.id)),
                    context,
                    outcome,
                )
            except TRANSPORT_ERRORS as exc:
                last_error = exc
                # Subscribe is unavailable — fall back to a poll.
                await asyncio.sleep(backoff)
                outcome.task = await client.get_task(GetTaskRequest(id=outcome.task.id))
                if outcome.task.status.state == TaskState.TASK_STATE_AUTH_REQUIRED:
                    raise A2AAuthRequiredError(outcome.task) from None
        except TRANSPORT_ERRORS as exc:
            last_error = exc
            await asyncio.sleep(backoff)

    if _needs_reconnect(outcome):
        raise A2AReconnectError(attempts=attempts, last_error=last_error)


def _needs_reconnect(outcome: StreamOutcome) -> bool:
    return (
        outcome.task is not None
        and outcome.task.status.state not in TERMINAL_TASK_STATES
        and not outcome.input_required
    )


def _read_input_prompt(task: Task, outcome: StreamOutcome) -> None:
    if not task.status.HasField("message"):
        return
    msg = task.status.message
    outcome.input_prompt = "".join(p.text for p in msg.parts if p.WhichOneof("content") == "text")
    outcome.input_metadata = message_metadata(msg)
