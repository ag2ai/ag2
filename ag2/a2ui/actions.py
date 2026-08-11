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
or invoke the agent yourself. Like an agent tool, the function may declare
``Depends(...)`` / ``Inject(...)`` parameters; they resolve against the agent's
``dependency_provider`` when the action runs (see :meth:`A2UIAction.run`).

A button the LLM draws that has **no** registered action still works: its click
is rewritten into a generic prompt so the agent can react (see
:func:`~ag2.a2ui.incoming.iter_incoming_prompts`). So "the agent reacts
to a click" needs no decorator at all; an action is only for running
deterministic server logic on click.
"""

from collections.abc import Callable, Iterable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, overload

from fast_depends.core import CallModel
from fast_depends.pydantic.schema import get_schema

from ag2.tools.final import FunctionParameters
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from ._types import JsonValue

if TYPE_CHECKING:
    from ag2.context import ConversationContext

    from .incoming import A2UIIncomingAction

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


@dataclass(slots=True, frozen=True)
class A2UIAction:
    """A clickable A2UI button bound to a server-side handler.

    Produced by :func:`a2ui_action` (not meant to be constructed directly).
    Carries the :class:`A2UIEventAction` declaration (so the LLM can render the
    button) and ``model`` — a ``fast_depends`` :class:`~fast_depends.core.CallModel`
    that the action runs (via :meth:`run`) with dependency injection and
    serializer-coerced ``event.context`` as input. (The handler's *return* is
    coerced to JSON by the caller — see
    :func:`~ag2.a2ui.server_action.run_server_action`.) It is
    deliberately **not** a tool: the agent never sees or calls it. Pass it in
    ``A2UIServer(actions=[...])``.
    """

    action: A2UIEventAction
    model: CallModel

    async def run(self, click: "A2UIIncomingAction", *, context: "ConversationContext") -> Any:
        """Execute this action's handler with dependency injection resolved.

        Mirrors :meth:`~ag2.tools.final.function_tool.FunctionTool.__call__`:
        the action is provider-agnostic, so the ``dependency_provider`` is taken
        from the live ``context`` at call time rather than baked into the action.
        The click's ``context`` supplies the handler's keyword arguments (coerced
        by the serializer), and generator dependencies are torn down with the
        ``AsyncExitStack`` after the handler returns.

        Args:
            click: The parsed incoming click; its ``context`` becomes the
                handler's keyword arguments.
            context: The conversation context supplying ``Depends``/``Inject``
                dependencies, variables, and the ``dependency_provider``.

        Returns:
            Whatever the handler returns (opaque user value; the caller maps it
            onto the wire).
        """
        async with AsyncExitStack() as stack:
            return await self.model.asolve(
                **(click.context | {CONTEXT_OPTION_NAME: context}),
                stack=stack,
                cache_dependencies={},
                dependency_provider=context.dependency_provider,
            )


def collect_action_declarations(actions: Iterable[object]) -> tuple[A2UIEventAction, ...]:
    """Return the :class:`A2UIEventAction` declaration of each :class:`A2UIAction`.

    Used to describe every clickable button to the LLM. Non-actions are ignored,
    so a mixed list works.
    """
    return tuple(a.action for a in actions if isinstance(a, A2UIAction))


def collect_server_actions(actions: Iterable[object]) -> dict[str, A2UIAction]:
    """Map action name → the :class:`A2UIAction` that runs on click.

    Used by the turn core / executor to run a click on the server without
    invoking the agent. The action (not its raw ``CallModel``) is returned so the
    object owns its own execution — :meth:`A2UIAction.run` runs the handler
    through ``fast_depends``, resolving ``Depends``/``Inject`` parameters and
    coercing ``event.context`` via the serializer. Non-actions are ignored.
    """
    return {a.action.name: a for a in actions if isinstance(a, A2UIAction)}


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
        return A2UIAction(
            action=action,
            model=call_model,
        )

    if function is not None:
        return make(function)
    return make


__all__ = (
    "A2UIAction",
    "A2UIEventAction",
    "a2ui_action",
    "collect_action_declarations",
    "collect_server_actions",
)
