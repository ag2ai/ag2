# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Best-effort conversion of an :class:`ag2.eval.Trace` into a probe-ready conversation."""

from dataclasses import dataclass

from ag2.eval import Trace
from ag2.events import (
    BaseEvent,
    ModelResponse,
    TaskCompleted,
    TaskFailed,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
)

from .types import Conversation, Turn

__all__ = (
    "TraceConversation",
    "conversation_from_trace",
)


@dataclass(frozen=True, slots=True)
class TraceConversation:
    """A conversation built from a trace, with each turn's source event.

    Attributes:
        conversation: The converted conversation.
        event_indices: For each turn, the index into ``trace.events`` it came from.
    """

    conversation: Conversation
    event_indices: tuple[int, ...]


def conversation_from_trace(
    trace: Trace,
    *,
    question: str,
    ground_truth: str,
    agent_name: str = "assistant",
) -> TraceConversation | None:
    """Turn the steps of ``trace`` into a conversation.

    A trace records one agent's stream, and its events carry no speaker, so
    the mapping is best effort:

    * a model response with text, and each tool call, is a turn by ``agent_name``;
    * a tool result or tool error is a turn by ``tool:<tool name>``;
    * a completed or failed sub-task is a turn by the sub-agent that ran it
      (``TaskCompleted.agent_name`` / ``TaskFailed.agent_name``).

    Other events (usage, streaming chunks, task start, ...) are skipped.

    Args:
        trace: The captured run.
        question: The task the run was solving.
        ground_truth: The expected answer.
        agent_name: Speaker of the traced agent's own turns.

    Returns:
        The conversation and each turn's event index, or ``None`` when the
        trace has no turns.
    """
    turns: list[Turn] = []
    indices: list[int] = []
    for index, event in enumerate(trace.events):
        turn = _turn(event, agent_name)
        if turn is not None:
            turns.append(turn)
            indices.append(index)
    if not turns:
        return None
    return TraceConversation(
        conversation=Conversation(question=question, ground_truth=ground_truth, history=tuple(turns)),
        event_indices=tuple(indices),
    )


def _turn(event: BaseEvent, agent_name: str) -> Turn | None:
    if isinstance(event, ToolErrorEvent):
        return Turn(name=f"tool:{event.name}", content=f"error: {event.error}")
    if isinstance(event, ToolResultEvent):
        return Turn(name=f"tool:{event.name}", content=_result_text(event))
    if isinstance(event, ToolCallEvent):
        return Turn(name=agent_name, content=f"calls {event.name}({event.arguments})")
    if isinstance(event, ModelResponse):
        return Turn(name=agent_name, content=event.content) if event.content else None
    if isinstance(event, TaskCompleted):
        return Turn(name=event.agent_name, content="" if event.result is None else str(event.result))
    if isinstance(event, TaskFailed):
        return Turn(name=event.agent_name, content=f"failed: {event.error}")
    return None


def _result_text(event: ToolResultEvent) -> str:
    return "\n".join(part.content if isinstance(part, TextInput) else repr(part) for part in event.result.parts)
