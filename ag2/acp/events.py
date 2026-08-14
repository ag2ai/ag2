# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACP-specific stream events.

These cover ACP ``session/update`` variants that have no existing AG2
equivalent (plan, mode change, available commands), plus the agent asking the
user a question (``elicitation/create``). Message chunks, thoughts, and tool
calls map onto existing AG2 events (see :mod:`.mappers`).
"""

from dataclasses import dataclass

from ag2.events import BaseEvent
from ag2.events.base import Field


@dataclass(frozen=True, slots=True)
class ACPPlanEntry:
    """A single entry of an ACP execution plan."""

    content: str
    status: str
    priority: str | None = None


class ACPPlan(BaseEvent):
    """ACP ``plan`` update — the agent's execution plan."""

    entries: list[ACPPlanEntry] = Field(default_factory=list, kw_only=False)


class ACPModeChange(BaseEvent):
    """ACP ``current_mode_update`` — the session's mode changed."""

    mode_id: str = Field(kw_only=False)


class ACPAvailableCommands(BaseEvent):
    """ACP ``available_commands_update`` — slash commands the agent advertises."""

    commands: list[str] = Field(default_factory=list, kw_only=False)


class ACPElicitation(BaseEvent):
    """ACP ``elicitation/create`` — the agent is asking the user a question.

    Emitted *before* the human is prompted, so an observer sees the question
    even when the answer comes from somewhere other than an interactive human
    (a declining policy, an absent HITL channel).

    Attributes:
        message: The agent's own description of what it needs.
        mode: ``"form"``, ``"url"``, or ``"other"`` for a mode this version does
            not understand (which is declined).
        url: The URL the user is directed to, for ``"url"`` mode.
        fields: The requested property names, in schema order, for ``"form"``
            mode. Names rather than the whole schema: the prompts the human
            answers are rendered by AG2, so what an observer needs is *which*
            fields were asked for, not how to render them again.
    """

    message: str = Field(kw_only=False)
    mode: str = Field(kw_only=False)
    url: str | None = None
    fields: list[str] = Field(default_factory=list)
