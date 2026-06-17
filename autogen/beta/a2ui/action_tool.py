# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Iterable
from typing import Any, overload

from fast_depends.core import CallModel
from fast_depends.pydantic.schema import get_schema

from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import FunctionParameters, FunctionTool
from autogen.beta.utils import CONTEXT_OPTION_NAME, build_model

from ._types import JsonValue
from .actions import A2UIEventAction

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
    """Build a placeholder ``event.context`` dict from a tool's parameter schema.

    Maps each top-level property to a type-tagged placeholder (e.g.
    ``{"time": "<string>"}``) so the LLM knows which keys a button click should
    send. This is illustrative only — an explicit ``example_context=`` overrides it.
    """
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    return {prop_name: _placeholder_for(prop_schema) for prop_name, prop_schema in properties.items()}


class A2UIActionTool(FunctionTool):
    """A :class:`FunctionTool` that is also exposed as a clickable A2UI button.

    Produced by the :func:`a2ui_action` decorator. It behaves exactly like a
    normal tool — the agent can call it during a turn — but it additionally
    carries an :class:`A2UIEventAction` (``tool_name`` set to its own name) so
    that a button click in the rendered UI routes back to this tool. Pass it in
    ``Agent(tools=[...])``; the transport discovers both the executable tool and
    its action.
    """

    __slots__ = ("action",)

    def __init__(
        self,
        model: CallModel,
        *,
        name: str,
        description: str,
        schema: FunctionParameters,
        action: A2UIEventAction,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        super().__init__(model, name=name, description=description, schema=schema, middleware=middleware)
        self.action = action


@overload
def a2ui_action(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
    middleware: Iterable[ToolMiddleware] = (),
) -> A2UIActionTool: ...


@overload
def a2ui_action(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
    middleware: Iterable[ToolMiddleware] = (),
) -> Callable[[Callable[..., Any]], A2UIActionTool]: ...


def a2ui_action(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    example_context: dict[str, JsonValue] | None = None,
    sync_to_thread: bool = True,
    middleware: Iterable[ToolMiddleware] = (),
) -> A2UIActionTool | Callable[[Callable[..., Any]], A2UIActionTool]:
    """Mark a tool function as a clickable A2UI button (a server ``event`` action).

    The decorated function becomes an :class:`A2UIActionTool` — a regular tool the
    agent can call, plus an :class:`A2UIEventAction` whose ``tool_name`` points
    back at the tool. When the LLM emits a button with ``action.event.name``
    matching this tool's name, a client click routes the ``event.context`` back
    as the tool's arguments. Pass the result directly in ``Agent(tools=[...])``.

    Only server ``event`` actions are produced here — a clickable button backed
    by a server tool. Purely client-side ``functionCall`` buttons (e.g.
    ``openUrl``) run on the renderer with no server round-trip, so they are not
    declared on the server; the LLM learns them from the catalog and the client's
    advertised capabilities.

    Args:
        function: The tool function (when used as a bare ``@a2ui_action``).
        name: Action/tool name. Defaults to the function name.
        description: Action/tool description. Defaults to the function docstring.
        example_context: Example ``event.context`` shown to the LLM. When omitted,
            a placeholder is derived from the function's parameter schema
            (e.g. ``{"time": "<string>"}``).
        sync_to_thread: Run a sync function in a worker thread. Forwarded to the tool.
        middleware: Tool middleware. Forwarded to the tool.

    Example::

        @a2ui_action(description="Schedule all posts for the given time")
        def schedule_posts(time: str) -> str: ...


        agent = Agent(tools=[schedule_posts])
    """

    def make(f: Callable[..., Any]) -> A2UIActionTool:
        call_model = build_model(f, sync_to_thread=sync_to_thread, serialize_result=False)
        action_name = name or f.__name__
        action_description = description or f.__doc__ or ""
        param_schema = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME,))
        ctx = example_context if example_context is not None else _derive_example_context(param_schema)
        action = A2UIEventAction(
            name=action_name,
            tool_name=action_name,
            description=action_description,
            example_context=ctx,
        )
        return A2UIActionTool(
            call_model,
            name=action_name,
            description=action_description,
            schema=param_schema,
            action=action,
            middleware=middleware,
        )

    if function is not None:
        return make(function)
    return make


__all__ = (
    "A2UIActionTool",
    "a2ui_action",
)
