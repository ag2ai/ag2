# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Map a server-side A2UI action's result onto the wire.

A server-side click never invokes the agent: the action runs on the server (via
:meth:`~ag2.a2ui.beta.actions.A2UIAction.run`, which resolves dependency
injection) with the click's ``event.context`` as keyword arguments, and whatever
it returns is turned into A2UI server→client messages per the spec:

- one server→client message (or a list of them, e.g. ``updateComponents`` /
  ``updateDataModel``) → emitted verbatim as a surface update;
- any other value → dropped with a debug log: an action is fire-and-forget on
  every version (``a2ui-core``'s v1.0 has no ``actionResponse``);
- a raised exception → logged.

The handler's return is coerced to JSON (``pydantic_core``) before any of the
above, so a Pydantic model / datetime / Enum / dataclass reaches the wire as
plain JSON — the same way the sibling AG-UI transport renders agent state.
"""

import logging
from typing import TYPE_CHECKING, Any, TypeGuard

from pydantic_core import to_jsonable_python

from ag2.context import ConversationContext, strip_reserved_variables
from ag2.stream import MemoryStream

from ._types import A2UIVersion, JsonObject, ServerToClientMessage
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
        "callRendererFunction",
        "agentFunctionResponse",
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

    *variables* come from the clicking client, so reserved keys are dropped.
    """
    return ConversationContext(
        MemoryStream(),
        dependencies=dict(agent._agent_dependencies),
        variables={
            **dict(agent._agent_variables),
            **strip_reserved_variables(variables or {}, source="an inbound A2UI click"),
        },
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
        click: The parsed incoming click (``name`` / ``context``).
        version: The wire ``version`` string to stamp on emitted messages.
        context: The conversation context supplying dependencies, variables, and
            the ``dependency_provider`` (see :func:`build_server_action_context`).

    Returns:
        The server→client messages to emit for this click (possibly empty).
    """
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
        # (logged), never tearing the turn down.
        result = to_jsonable_python(result)
    except Exception:  # noqa: BLE001 - a handler failure must not tear down the turn
        logger.exception("A2UI server action %r failed", click.name)
        return []

    return _result_to_messages(result, action_name=click.name, version=version)


def _result_to_messages(result: object, *, action_name: str, version: A2UIVersion) -> list[ServerToClientMessage]:
    # A2UI message(s): emit verbatim as surface updates.
    if _is_message(result):
        return [_stamp(result, version)]
    if isinstance(result, list) and result and all(_is_message(item) for item in result):
        return [_stamp(item, version) for item in result]

    # Fire-and-forget. A non-None result with nowhere to go is dropped.
    if result is not None:
        logger.debug("A2UI server action %r returned a value that is not an A2UI message; dropping it.", action_name)
    return []


__all__ = ("build_server_action_context", "run_server_action")
