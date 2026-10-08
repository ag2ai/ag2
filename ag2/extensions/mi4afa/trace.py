# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Best-effort conversion of an :class:`ag2.eval.Trace` into a probe-ready conversation."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from ag2.tools.final import Toolkit
from ag2.tools.subagents.subagent_tool import SubagentTool
from ag2.tools.tool import Tool

from .types import Conversation, Turn

if TYPE_CHECKING:
    from ag2 import Agent

__all__ = (
    "TraceConversation",
    "conversation_from_trace",
    "subagent_names",
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
    subagents: Mapping[str, str] | None = None,
) -> TraceConversation | None:
    """Turn the steps of ``trace`` into a conversation.

    A trace records one agent's stream, and its events carry no speaker, so
    the mapping is best effort:

    * a model response with text, and each tool call, is a turn by ``agent_name``;
    * a tool result or tool error is a turn by ``tool:<tool name>``, or by the
      sub-agent ``subagents`` maps that tool to;
    * a completed or failed sub-task is a turn by the sub-agent that ran it
      (``TaskCompleted.agent_name`` / ``TaskFailed.agent_name``).

    Other events (usage, streaming chunks, task start, ...) are skipped.

    A trace rebuilt from spans, as :func:`ag2.eval.run_agent` and the OTEL
    trace sources produce, has no sub-task events: a sub-agent's work appears
    only as its delegation tool's result, so pass ``subagents`` to name it.

    Args:
        trace: The captured run.
        question: The task the run was solving.
        ground_truth: The expected answer.
        agent_name: Speaker of the traced agent's own turns.
        subagents: Delegation tool name to the sub-agent it runs, such as
            :func:`subagent_names` returns.

    Returns:
        The conversation and each turn's event index, or ``None`` when the
        trace has no turns.
    """
    subagents = subagents or {}
    turns: list[Turn] = []
    indices: list[int] = []
    for index, event in enumerate(trace.events):
        turn = _turn(event, agent_name, subagents)
        if turn is not None:
            turns.append(turn)
            indices.append(index)
    if not turns:
        return None
    return TraceConversation(
        conversation=Conversation(question=question, ground_truth=ground_truth, history=tuple(turns)),
        event_indices=tuple(indices),
    )


def subagent_names(agent: "Agent") -> dict[str, str]:
    """Map each delegation tool of ``agent`` to the name of the sub-agent it runs.

    Covers tools made with :meth:`ag2.Agent.as_tool` or ``subagent_tool``,
    under their default or a custom name, including inside toolkits. A
    background sub-agent tool returns a task id rather than the sub-agent's
    answer, so it is left out, as are tools passed to ``ask`` instead of
    ``agent.tools``.
    """
    return dict(_subagent_names(agent.tools))


def _subagent_names(tools: Iterable[Tool]) -> Iterable[tuple[str, str]]:
    for tool in tools:
        if isinstance(tool, SubagentTool):
            yield tool.name, tool.agent.name
        elif isinstance(tool, Toolkit):
            yield from _subagent_names(tool.tools)


def _turn(event: BaseEvent, agent_name: str, subagents: Mapping[str, str]) -> Turn | None:
    if isinstance(event, ToolErrorEvent):
        return Turn(name=_tool_speaker(event.name, subagents), content=f"error: {event.error}")
    if isinstance(event, ToolResultEvent):
        return Turn(name=_tool_speaker(event.name, subagents), content=_result_text(event))
    if isinstance(event, ToolCallEvent):
        return Turn(name=agent_name, content=f"calls {event.name}({event.arguments})")
    if isinstance(event, ModelResponse):
        return Turn(name=agent_name, content=event.content) if event.content else None
    if isinstance(event, TaskCompleted):
        return Turn(name=event.agent_name, content="" if event.result is None else str(event.result))
    if isinstance(event, TaskFailed):
        return Turn(name=event.agent_name, content=f"failed: {event.error}")
    return None


def _tool_speaker(tool_name: str | None, subagents: Mapping[str, str]) -> str:
    if tool_name is not None and tool_name in subagents:
        return subagents[tool_name]
    return f"tool:{tool_name}"


def _result_text(event: ToolResultEvent) -> str:
    return "\n".join(part.content if isinstance(part, TextInput) else repr(part) for part in event.result.parts)
