# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""ACP-specific stream events.

These cover ACP ``session/update`` variants that have no existing AG2
equivalent (plan, mode change, available commands), plus the agent asking the
user a question (``elicitation/create``) or for permission
(``session/request_permission``). Message chunks, thoughts, and tool calls map
onto existing AG2 events (see :mod:`.mappers`).
"""

from dataclasses import dataclass
from typing import Any, Literal

from ag2.events import BaseEvent, ToolApprovalRequest
from ag2.events.base import Field

# Mirrors of ACP's closed enums, kept here so the events need no `acp` import.
PermissionOptionKind = Literal["allow_once", "allow_always", "reject_once", "reject_always"]
ToolKind = Literal["read", "edit", "delete", "move", "search", "execute", "think", "fetch", "switch_mode", "other"]


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


@dataclass(frozen=True, slots=True)
class ACPPermissionOption:
    """One choice the agent offers in a permission request.

    Attributes:
        option_id: The ``ApprovalChoiceMessage`` content that selects this option.
        name: The agent's label. It, not ``kind``, says what an ``*_always``
            option covers (for example "Allow for this session").
        kind: ``allow_once``, ``allow_always``, ``reject_once`` or
            ``reject_always`` — a hint for how to present the option.
    """

    option_id: str
    name: str
    kind: PermissionOptionKind


@dataclass(frozen=True, slots=True)
class ACPToolLocation:
    """A file the tool call the permission is for affects."""

    path: str
    line: int | None = None


class ACPPermissionRequest(ToolApprovalRequest):
    """ACP ``session/request_permission`` — the agent asks before running a tool call.

    A :class:`~ag2.events.ToolApprovalRequest`, so anything that handles tool
    approvals handles this one, and carries the request as the agent sent it so
    a hook or UI can offer the agent's own options. ``content`` is the tool
    call's agent-authored title, and ``choices`` holds the options as
    transport-neutral choices (``option_id`` and ``name``).

    Answer with an :class:`~ag2.events.ApprovalChoiceMessage` holding the
    ``option_id`` of the chosen option. A plain answer is read as a yes or no
    in words, never as an id, and is never widened: ``yes`` selects the
    one-time allow option, ``always`` the remembered one, anything else rejects.

    Attributes:
        options: The agent's options, in the agent's order.
        kind: The tool call's ACP ``ToolKind`` (``"execute"``, ``"edit"``, ...),
            or ``None`` if the agent sent none.
        locations: The files the tool call affects.
        raw_input: The tool call's raw input as the agent sent it.
    """

    options: list[ACPPermissionOption] = Field(default_factory=list)
    kind: ToolKind | None = None
    locations: list[ACPToolLocation] = Field(default_factory=list)
    raw_input: Any = None
