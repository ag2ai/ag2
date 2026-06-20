# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2UI actions — the single concept behind a clickable button.

An action is declared with :func:`a2ui_action` and registered via
``A2UIServer(actions=[...])`` (or ``A2UIAgentExecutor(actions=[...])``). The
agent renders the button (the action is declared to the LLM so it knows the
button exists and what ``context`` to send), but a **click runs the function on
the server** — it is *not* an agent tool and never enters the agent's tool
machinery. Inside the function you can do anything: hit a backend, call a tool,
or invoke the agent yourself.

A button the LLM draws that has **no** registered action still works: its click
is rewritten into a generic prompt so the agent can react (see
:func:`~autogen.beta.a2ui.incoming.iter_incoming_prompts`). So "the agent reacts
to a click" needs no decorator at all; an action is only for running
deterministic server logic on click.
"""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, overload

from fast_depends.pydantic.schema import get_schema

from autogen.beta.tools.final import FunctionParameters
from autogen.beta.utils import CONTEXT_OPTION_NAME, _to_async, build_model

from ._types import JsonValue

# JSON-schema ``type`` → placeholder shown to the LLM as the expected
# ``event.context`` value. Used only when no explicit ``example_context`` is given.
# Scalar placeholders are immutable; "array"/"object" are produced fresh per use
# (see ``_placeholder_for``) so the returned example context never aliases shared state.
_TYPE_PLACEHOLDERS: dict[str, JsonValue] = {
    "string": "<string>",
    "integer": "<integer>",
    "number": "<number>",
    "boolean": "<boolean>",
}


@dataclass(slots=True, frozen=True)
class A2UIEventAction:
    """A server ``event`` action — the declaration behind a clickable button.

    Produced internally by the :func:`a2ui_action` decorator (it is not meant to
    be constructed directly). It is used to (1) describe the button in the system
    prompt so the LLM can render it, and (2) recognize an incoming click by name
    so its server handler runs instead of routing the click to the agent.

    Args:
        name: Action identifier; matches the ``event.name`` in the button's
            action definition.
        description: Human-readable description, injected into the system prompt
            so the LLM knows the action exists.
        example_context: Example ``event.context`` dict shown to the LLM,
            matching the handler's parameter names.
    """

    name: str
    description: str = ""
    example_context: dict[str, JsonValue] | None = None

    # Discriminator as a fixed instance field (``init=False`` so callers can't
    # set it), kept for forward-compatibility with additional action kinds.
    action_type: Literal["event"] = field(default="event", init=False)


def _placeholder_for(prop_schema: object) -> JsonValue:
    """Map one JSON-schema property to an illustrative ``event.context`` value.

    Handles plain ``"type"`` schemas, ``array``/``object`` (fresh container each
    call), and the ``anyOf``/``oneOf`` form Pydantic emits for ``Optional[...]``
    (first non-null branch wins). Anything unrecognized falls back to ``"<value>"``.
    """
    if not isinstance(prop_schema, dict):
        return "<value>"
    prop_type = prop_schema.get("type")
    if prop_type == "array":
        return []
    if prop_type == "object":
        return {}
    if isinstance(prop_type, str):
        return _TYPE_PLACEHOLDERS.get(prop_type, "<value>")
    for branch_key in ("anyOf", "oneOf"):
        branches = prop_schema.get(branch_key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
                    return _placeholder_for(branch)
    return "<value>"


def _derive_example_context(schema: FunctionParameters) -> dict[str, JsonValue]:
    """Build a placeholder ``event.context`` dict from a handler's parameter schema.

    Maps each top-level property to a type-tagged placeholder (e.g.
    ``{"time": "<string>"}``) so the LLM knows which keys a button click should
    send. This is illustrative only — an explicit ``example_context=`` overrides it.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    return {prop_name: _placeholder_for(prop_schema) for prop_name, prop_schema in properties.items()}


# A server-side action handler: an async callable invoked with the click's
# resolved ``event.context`` as keyword arguments.
ServerActionHandler = Callable[..., Awaitable[Any]]


@dataclass(slots=True, frozen=True)
class A2UIAction:
    """A clickable A2UI button bound to a server-side handler.

    Produced by :func:`a2ui_action` (not meant to be constructed directly).
    Carries the :class:`A2UIEventAction` declaration (so the LLM can render the
    button) and the async ``handler`` that runs on click. It is deliberately
    **not** a tool: the agent never sees or calls it. Pass it in
    ``A2UIServer(actions=[...])``.
    """

    action: A2UIEventAction
    handler: ServerActionHandler


def collect_action_declarations(actions: Iterable[object]) -> tuple[A2UIEventAction, ...]:
    """Return the :class:`A2UIEventAction` declaration of each :class:`A2UIAction`.

    Used to describe every clickable button to the LLM. Non-actions are ignored,
    so a mixed list works.
    """
    return tuple(a.action for a in actions if isinstance(a, A2UIAction))


def collect_server_handlers(actions: Iterable[object]) -> dict[str, ServerActionHandler]:
    """Map action name → handler for each :class:`A2UIAction`.

    Used by the turn core / executor to run a click on the server without
    invoking the agent. Non-actions are ignored.
    """
    return {a.action.name: a.handler for a in actions if isinstance(a, A2UIAction)}


@overload
def a2ui_action(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
) -> A2UIAction: ...


@overload
def a2ui_action(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], A2UIAction]: ...


def a2ui_action(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
) -> A2UIAction | Callable[[Callable[..., Any]], A2UIAction]:
    """Mark a function as a clickable, **server-side** A2UI button.

    The decorated function becomes an :class:`A2UIAction`. The button is declared
    to the LLM (so it can render it), but a click runs this function on the
    server with the click's ``event.context`` as keyword arguments — the agent is
    **not** invoked. What the function returns is mapped to the client per the
    A2UI spec:

    - one A2UI server→client message, or a list of them (e.g. ``updateComponents``
      / ``updateDataModel``) → sent to the renderer as a surface update (works on
      every protocol version);
    - any other JSON value → returned as an ``actionResponse`` **only** when the
      client requested one (``wantResponse`` + ``actionId``, v1.0); otherwise it
      is fire-and-forget.

    Register the result with ``A2UIServer(actions=[...])``. A button the LLM draws
    with no registered action still works — its click is rewritten into a generic
    prompt so the agent can react — so use this decorator only when a click should
    run deterministic server logic.

    Args:
        function: The function (when used as a bare ``@a2ui_action``).
        name: Action name. Defaults to the function name.
        description: Action description. Defaults to the function docstring.
        example_context: Example ``event.context`` shown to the LLM. When omitted,
            a placeholder is derived from the function's parameter schema
            (e.g. ``{"good_id": "<string>"}``).
        sync_to_thread: Run a sync function in a worker thread.

    Example::

        @a2ui_action(description="Add this item to the cart")
        def add_to_basket(good_id: str) -> dict:
            count = cart.add(good_id)
            return {"updateDataModel": {"surfaceId": "cart", "path": "/count", "value": count}}


        server = A2UIServer(agent, actions=[add_to_basket], transport=...)
    """

    def make(f: Callable[..., Any]) -> A2UIAction:
        # build_model derives the parameter schema (and validates the signature),
        # which drives example_context derivation.
        call_model = build_model(f, sync_to_thread=sync_to_thread, serialize_result=False)
        action_name = name or f.__name__
        action_description = description or f.__doc__ or ""
        param_schema = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME,))
        ctx = example_context if example_context is not None else _derive_example_context(param_schema)
        action = A2UIEventAction(
            name=action_name,
            description=action_description,
            example_context=ctx,
        )
        return A2UIAction(action=action, handler=_to_async(f, sync_to_thread=sync_to_thread))

    if function is not None:
        return make(function)
    return make


__all__ = (
    "A2UIAction",
    "A2UIEventAction",
    "ServerActionHandler",
    "a2ui_action",
    "collect_action_declarations",
    "collect_server_handlers",
)
