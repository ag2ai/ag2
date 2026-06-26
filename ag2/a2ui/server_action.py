# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Map a server-side A2UI action's result onto the wire.

A server-side click never invokes the agent: the action runs on the server (via
:meth:`~ag2.a2ui.actions.A2UIAction.run`, which resolves dependency
injection) with the click's ``event.context`` as keyword arguments, and whatever
it returns is turned into A2UI server→client messages per the spec:

- one server→client message (or a list of them, e.g. ``updateComponents`` /
  ``updateDataModel``) → emitted verbatim as a surface update (every version);
- any other value → wrapped in an ``actionResponse`` **only** when the
  client asked for one (``wantResponse`` + ``actionId``, v1.0) — otherwise
  dropped with a debug log (no pre-v1.0 wire for it);

The handler's return is coerced to JSON (``pydantic_core``) before any of the
above, so a Pydantic model / datetime / Enum / dataclass reaches the wire as
plain JSON — the same way the sibling AG-UI transport renders agent state.
- ``None`` / no response requested → fire-and-forget;
- a raised exception → an ``actionResponse`` error when a response was
  requested, else logged.
"""

import logging
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from pydantic_core import to_jsonable_python

from ag2.context import ConversationContext
from ag2.stream import MemoryStream

from ._types import A2UIVersion, JsonObject, JsonValue, ServerToClientMessage
from .actions import A2UIAction
from .incoming import A2UIIncomingAction

if TYPE_CHECKING:
    from ag2.agent import Agent

logger = logging.getLogger(__name__)

# Top-level keys that identify a dict as an A2UI server→client message rather
# than a plain return value. Kept in sync with ``ServerToClientMessage``.
_MESSAGE_KEYS = frozenset(
    {
        "createSurface",
        "updateComponents",
        "updateDataModel",
        "deleteSurface",
        "callFunction",
        "actionResponse",
    },
)


def _is_message(value: object) -> TypeGuard[JsonObject]:
    """True if ``value`` looks like an A2UI server→client message envelope.

    A :class:`~typing.TypeGuard` so a ``True`` result narrows the (otherwise
    opaque) handler return to a JSON object at the call site.
    """
    return isinstance(value, dict) and any(k in value for k in _MESSAGE_KEYS)


def _stamp(message: dict[str, Any], version: A2UIVersion) -> ServerToClientMessage:
    """Ensure a handler-returned message carries the wire ``version``."""
    if "version" not in message:
        message = {"version": version, **message}
    return message  # type: ignore[return-value]


def _action_response(
    action_id: str,
    version: A2UIVersion,
    *,
    value: JsonValue = None,
    error: dict[str, str] | None = None,
) -> ServerToClientMessage:
    body: dict[str, Any] = {"error": error} if error is not None else {"value": value}
    message: dict[str, Any] = {"version": version, "actionId": action_id, "actionResponse": body}
    return message  # type: ignore[return-value]


def build_server_action_context(
    agent: "Agent",
    *,
    variables: dict[str, Any] | None = None,
) -> ConversationContext:
    """Build the :class:`ConversationContext` a server action runs against.

    A click runs without invoking the agent, so it has no turn context of its
    own. This mirrors what the agent itself receives — the agent's dependencies
    (for ``Inject``), merged variables (for ``Variable``), and the agent's
    ``dependency_provider`` (for ``Depends`` resolution and
    ``dependency_provider.override(...)``) — over a throwaway stream, so a
    handler's ``Depends``/``Inject`` parameters resolve exactly like a tool's.
    """
    return ConversationContext(
        MemoryStream(),
        dependencies=dict(agent._agent_dependencies),
        variables={**dict(agent._agent_variables), **(variables or {})},
        dependency_provider=agent.dependency_provider,
    )


async def run_server_action(
    action: A2UIAction,
    click: A2UIIncomingAction,
    *,
    version: A2UIVersion,
    context: ConversationContext,
) -> list[ServerToClientMessage]:
    """Run a server-side action and map its result to A2UI messages.

    The DI execution lives on the action itself (:meth:`A2UIAction.run`, the same
    path as an agent tool); this function owns only the wire mapping — turning the
    handler's return (or failure) into A2UI server→client messages.

    Args:
        action: The :class:`A2UIAction` to run. :meth:`A2UIAction.run` solves it
            against ``context`` so ``Depends``/``Inject`` parameters resolve and
            ``event.context`` is coerced by the serializer.
        click: The parsed incoming click (``name`` / ``context`` /
            ``response_request``).
        version: The wire ``version`` string to stamp on emitted messages and
            the protocol version gating ``actionResponse`` (v1.0 only).
        context: The conversation context supplying dependencies, variables, and
            the ``dependency_provider`` (see :func:`build_server_action_context`).

    Returns:
        The server→client messages to emit for this click (possibly empty).
    """
    wants_response = click.response_request is not None
    can_respond = wants_response and version == "v1.0"
    if wants_response and not can_respond:
        # ``actionResponse`` does not exist before v1.0; a client that set
        # wantResponse on an older channel cannot be answered in-protocol.
        logger.warning(
            "A2UI server action %r requested a response, but actionResponse requires v1.0 (have %s); "
            "the handler result will not be returned to the client.",
            click.name,
            version,
        )
    action_id = click.response_request.action_id if click.response_request is not None else ""

    try:
        result = await action.run(click, context=context)
        # Coerce the handler's (opaque) return into an embeddable JSON value, so a
        # Pydantic model / datetime / Enum / dataclass becomes JSON-native before
        # it reaches ``json.dumps`` (REST), the AG-UI encoder, or the A2A
        # ``DataPart``. This mirrors the sibling AG-UI transport, which renders
        # agent state onto the wire the same way (``ag_ui/stream.py``'s
        # ``_encode_context``); ``pydantic_core`` is the same engine the agent
        # serializer uses for text wire slots. Kept inside the ``try`` so a
        # serialization failure is treated like any other handler failure below
        # (surfaced as an ``actionResponse`` error or logged), never tearing the
        # turn down.
        result = to_jsonable_python(result)
    except Exception as e:  # noqa: BLE001 - a handler failure must not tear down the turn
        logger.exception("A2UI server action %r failed", click.name)
        if can_respond:
            return [_action_response(action_id, version, error={"code": "ACTION_FAILED", "message": str(e)})]
        return []

    return _result_to_messages(
        result, action_name=click.name, version=version, can_respond=can_respond, action_id=action_id
    )


def _result_to_messages(
    result: object,
    *,
    action_name: str,
    version: A2UIVersion,
    can_respond: bool,
    action_id: str,
) -> list[ServerToClientMessage]:
    # A2UI message(s): emit verbatim as surface updates.
    if _is_message(result):
        return [_stamp(result, version)]
    if isinstance(result, list) and result and all(_is_message(item) for item in result):
        return [_stamp(item, version) for item in result]

    # A correlated reply was requested → wrap the (possibly None) value. The
    # handler return is opaque (user code); the spec says it is a JSON value
    # here, so the cast marks that boundary before it goes on the wire.
    if can_respond:
        return [_action_response(action_id, version, value=cast(JsonValue, result))]

    # Fire-and-forget. A non-None result with nowhere to go is dropped.
    if result is not None:
        logger.debug(
            "A2UI server action %r returned a value but the client did not request a response; dropping it.",
            action_name,
        )
    return []


__all__ = ("build_server_action_context", "run_server_action")
