# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Resolve ACP ``session/request_permission`` requests to a selected option.

ACP leaves the choice to the user: the client shows the agent's options and
answers with the ``option_id`` of the one picked. Under ``"ask"`` the request
goes to the human as an :class:`~.events.ACPPermissionRequest` carrying those
options, and the answer selects one. ``"auto"`` / ``"deny"`` select the
one-time option, and the remembered one only when the agent offers no other.

Returns the ``option_id`` to select, or ``None`` when no offered option fits;
the bridge then answers ``cancelled``.
"""

from typing import TYPE_CHECKING

from acp import schema

from ag2.events import ApprovalChoice

from .events import ACPPermissionOption, ACPPermissionRequest, ACPToolLocation

if TYPE_CHECKING:
    from ag2.context import ConversationContext

    from .config import PermissionPolicy

# Plain approval answers, for a hook or UI that answers any tool approval
# (`approval_required`'s words; AG-UI's approve button arrives as "y").
_ONCE = {"y", "yes", "1", "ok", "allow", "approve", "allow_once"}
_ALWAYS = {"always", "allow_always"}


async def resolve_permission_option_id(
    policy: "PermissionPolicy",
    options: list[schema.PermissionOption],
    tool_call: schema.ToolCallUpdate,
    context: "ConversationContext | None",
) -> str | None:
    """Return the ``option_id`` to select, or ``None`` if no option fits.

    Args:
        policy: ``"ask"`` | ``"auto"`` | ``"deny"``.
        options: ACP permission options (``option_id``, ``kind``, ``name``).
        tool_call: The tool call the permission is for.
        context: The conversation context used to ask a human, or ``None``.
    """
    if policy == "auto":
        return _first_of_kind(options, "allow_once", "allow_always")
    if policy == "deny" or context is None:
        return _first_of_kind(options, "reject_once", "reject_always")

    answer = await context.ask(permission_request(options, tool_call))
    return _selected(answer.strip(), options)


def permission_request(
    options: list[schema.PermissionOption], tool_call: schema.ToolCallUpdate
) -> ACPPermissionRequest:
    """The human-input request for one ``session/request_permission``."""
    return ACPPermissionRequest(
        tool_call.title or "Allow this action?",
        tool_call_id=tool_call.tool_call_id,
        options=[ACPPermissionOption(o.option_id, o.name, o.kind) for o in options],
        # The same options as transport-neutral choices, so AG-UI can offer one button each.
        choices=[ApprovalChoice(o.option_id, o.name) for o in options],
        kind=tool_call.kind,
        locations=[ACPToolLocation(loc.path, loc.line) for loc in tool_call.locations or []],
        raw_input=tool_call.raw_input,
    )


def _selected(answer: str, options: list[schema.PermissionOption]) -> str | None:
    """The option the answer picks; anything unrecognised rejects.

    Plain approval words are read before ids, so an agent cannot make ``y``
    mean ``allow_always`` by giving that option the id ``"y"``.
    """
    word = answer.lower()
    if word in _ALWAYS:
        # With nothing remembered on offer, grant this call only: less than asked, never more.
        return _first_of_kind(options, "allow_always", "allow_once", "reject_once", "reject_always")
    if word in _ONCE:
        # Never widened to allow_always: with nothing one-time to allow, refuse.
        return _first_of_kind(options, "allow_once", "reject_once", "reject_always")
    for o in options:
        if o.option_id == answer:
            return o.option_id
    return _first_of_kind(options, "reject_once", "reject_always")


def _first_of_kind(options: list[schema.PermissionOption], *kinds: str) -> str | None:
    """The first offered option of the earliest kind in ``kinds``.

    ``kinds`` is a preference order: the agent's option order never decides
    between a one-time and a remembered option.
    """
    for kind in kinds:
        for o in options:
            if o.kind == kind:
                return o.option_id
    return None
