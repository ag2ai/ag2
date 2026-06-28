# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACP-specific stream events.

These cover ACP ``session/update`` variants that have no existing AG2 beta
equivalent (plan, mode change, available commands). Message chunks, thoughts,
and tool calls map onto existing beta events (see :mod:`.mappers`).
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
