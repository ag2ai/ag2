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
- any other value → dropped with a debug log: an ``action`` is fire-and-forget.

A renderer that wants the value back asks with a v1.0 ``callAgentFunction``
instead. That runs the same registered action (its ``args`` become the keyword
arguments) and always answers with an ``agentFunctionResponse`` that echoes the
call's ``functionCallId``: the returned value on success, or an ``error``
(``UNKNOWN_FUNCTION`` when nothing is registered under that name,
``INVALID_FUNCTION_CALL`` when the arguments do not fit, ``FUNCTION_FAILED``
when the handler raised). Messages the handler returns are emitted too, ahead
of the response.

The handler's return is coerced to JSON (``pydantic_core``) before any of the
above, so a Pydantic model / datetime / Enum / dataclass reaches the wire as
plain JSON — the same way the sibling AG-UI transport renders agent state.
"""

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, TypeGuard

from fast_depends.pydantic.schema import get_schema
from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from ag2.context import ConversationContext, strip_reserved_variables
from ag2.stream import MemoryStream
from ag2.utils import CONTEXT_OPTION_NAME

from ._types import A2UIVersion, JsonObject, JsonValue, ServerToClientMessage, server_to_client_payload_keys
from .actions import A2UIAction
from .incoming import (
    A2UIIncomingAction,
    A2UIIncomingActionResult,
    A2UIIncomingFunctionCall,
    A2UIIncomingFunctionCallResult,
    A2UIIncomingParseResult,
)

if TYPE_CHECKING:
    from ag2.agent import Agent

logger = logging.getLogger(__name__)


def _is_message(value: JsonValue) -> TypeGuard[JsonObject]:
    """True if ``value`` looks like an A2UI server→client message envelope.

    A :class:`~typing.TypeGuard` so a ``True`` result narrows the (otherwise
    opaque) handler return to a JSON object at the call site.
    """
    return isinstance(value, dict) and not server_to_client_payload_keys().isdisjoint(value)


def _stamp(message: JsonObject, version: A2UIVersion) -> ServerToClientMessage:
    """Ensure a handler-returned message carries the wire ``version``."""
    if "version" not in message:
        message = {"version": version, **message}
    return message  # type: ignore[return-value]


def _function_response(
    function_call_id: str,
    version: A2UIVersion,
    *,
    value: JsonValue = None,
    error: dict[str, str] | None = None,
) -> ServerToClientMessage:
    body: dict[str, Any] = {"functionCallId": function_call_id}
    body.update({"error": error} if error is not None else {"value": value})
    message: dict[str, Any] = {"version": version, "agentFunctionResponse": body}
    return message  # type: ignore[return-value]


def _invalid_function_arguments(call: A2UIIncomingFunctionCall, version: A2UIVersion) -> ServerToClientMessage:
    return _function_response(
        call.function_call_id,
        version,
        error={"code": "INVALID_FUNCTION_CALL", "message": f"Invalid arguments for function {call.call!r}."},
    )


def _has_unknown_arguments(action: A2UIAction, arguments: JsonObject) -> bool:
    if action.model.kwargs_name is not None:
        return False
    properties = get_schema(action.model, exclude=(CONTEXT_OPTION_NAME,)).get("properties", {})
    return any(name not in properties for name in arguments)


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
    """Run a server-side action for a click and map its result to A2UI messages.

    The DI execution lives on the action itself (:meth:`A2UIAction.run`, the same
    path as an agent tool); this function owns only the wire mapping — turning the
    handler's return into A2UI server→client messages. A click is fire-and-forget,
    so a failing handler is logged and answered with nothing.

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
        result = await _run_handler(action, click, context)
    except Exception:  # noqa: BLE001 - a handler failure must not tear down the turn
        logger.exception("A2UI server action %r failed", click.name)
        return []

    messages = _messages_of(result, version)
    if not messages and result is not None:
        logger.debug(
            "A2UI server action %r returned a value, but an action has no response channel; dropping it. "
            "Use a callAgentFunction to return a value to the renderer.",
            click.name,
        )
    return messages


async def run_agent_function(
    action: A2UIAction | None,
    call: A2UIIncomingFunctionCall,
    *,
    version: A2UIVersion,
    catalog_id: str,
    context: ConversationContext,
) -> list[ServerToClientMessage]:
    """Answer a renderer's ``callAgentFunction`` (v1.0) with an ``agentFunctionResponse``.

    Args:
        action: The registered :class:`A2UIAction` named by the call, or ``None``
            when nothing is registered under that name (answered ``UNKNOWN_FUNCTION``).
        call: The parsed incoming call; its ``args`` become the handler's keyword arguments.
        version: The wire ``version`` to stamp on emitted messages. ``callAgentFunction``
            only exists in v1.0, so any other version is logged and answered with nothing.
        catalog_id: The active catalog. An explicit different catalog must not
            select a local action merely because its function name matches.
        context: The conversation context supplying dependencies, variables, and
            the ``dependency_provider`` (see :func:`build_server_action_context`).

    Returns:
        Any messages the handler returned, then the response correlated by
        ``call.function_call_id`` — always exactly one, the renderer is waiting on it.
    """
    if version != "v1.0":
        logger.warning(
            "A2UI callAgentFunction %r needs protocol v1.0 (have %s); it cannot be answered in-protocol.",
            call.call,
            version,
        )
        return []

    if action is None or call.catalog_id not in (None, catalog_id):
        return [
            _function_response(
                call.function_call_id,
                version,
                error={"code": "UNKNOWN_FUNCTION", "message": f"Unknown function {call.call!r}."},
            )
        ]

    if not isinstance(call.args, dict) or _has_unknown_arguments(action, call.args):
        return [_invalid_function_arguments(call, version)]

    try:
        result = await _run_handler(action, call, context)
    except ValidationError as e:
        logger.warning("A2UI function %r called with invalid arguments: %s", call.call, e)
        return [_invalid_function_arguments(call, version)]
    except Exception:  # noqa: BLE001 - a handler failure must not tear down the turn
        logger.exception("A2UI function %r failed", call.call)
        return [
            _function_response(
                call.function_call_id,
                version,
                error={"code": "FUNCTION_FAILED", "message": f"Function {call.call!r} failed."},
            )
        ]

    messages = _messages_of(result, version)
    # Messages are the handler's surface updates; there is no separate value to return.
    value = None if messages else result
    return [*messages, _function_response(call.function_call_id, version, value=value)]


async def run_client_interactions(
    interactions: Iterable[A2UIIncomingParseResult],
    server_actions: Mapping[str, A2UIAction],
    agent: "Agent",
    *,
    variables: dict[str, Any] | None,
    version: A2UIVersion,
    catalog_id: str,
) -> tuple[list[ServerToClientMessage], bool]:
    """Run the interactions the server answers itself, without invoking the agent.

    That is every click on a registered action, and every ``callAgentFunction``
    (answered even when nothing is registered, since the renderer is waiting).
    Shared by every transport so they agree on what is handled where.

    Returns:
        The messages to emit, and whether any interaction was handled here — a turn
        whose only content was handled interactions needs no agent run.
    """
    messages: list[ServerToClientMessage] = []
    handled = False
    context: ConversationContext | None = None
    for interaction in interactions:
        if isinstance(interaction, A2UIIncomingActionResult):
            action = server_actions.get(interaction.action.name)
            if action is None:
                continue
            handled = True
            if context is None:
                context = build_server_action_context(agent, variables=variables)
            messages.extend(await run_server_action(action, interaction.action, version=version, context=context))
        elif isinstance(interaction, A2UIIncomingFunctionCallResult):
            call = interaction.function_call
            if context is None:
                context = build_server_action_context(agent, variables=variables)
            answer = await run_agent_function(
                server_actions.get(call.call), call, version=version, catalog_id=catalog_id, context=context
            )
            # Before v1.0 there is no reply to send: leave the turn to the agent.
            handled = handled or bool(answer)
            messages.extend(answer)
    return messages, handled


async def _run_handler(
    action: A2UIAction,
    invocation: A2UIIncomingAction | A2UIIncomingFunctionCall,
    context: ConversationContext,
) -> JsonValue:
    """Run the handler and coerce its (opaque) return into an embeddable JSON value.

    So a Pydantic model / datetime / Enum / dataclass becomes JSON-native before
    it reaches ``json.dumps`` (REST), the AG-UI encoder, or the A2A ``DataPart``.
    This mirrors the sibling AG-UI transport, which renders agent state onto the
    wire the same way (``ag_ui/stream.py``'s ``_encode_context``); ``pydantic_core``
    is the same engine the agent serializer uses for text wire slots. Coercion sits
    with the handler call so a serialization failure is treated like any other
    handler failure by the callers, never tearing the turn down.
    """
    result: JsonValue = to_jsonable_python(await action.run(invocation, context=context))
    return result


def _messages_of(result: JsonValue, version: A2UIVersion) -> list[ServerToClientMessage]:
    """The handler return as A2UI message(s) to emit verbatim, or ``[]`` when it is a plain value."""
    if _is_message(result):
        return [_stamp(result, version)]
    if isinstance(result, list) and result:
        items = [item for item in result if _is_message(item)]
        if len(items) == len(result):
            return [_stamp(item, version) for item in items]
    return []


__all__ = (
    "build_server_action_context",
    "run_agent_function",
    "run_client_interactions",
    "run_server_action",
)
