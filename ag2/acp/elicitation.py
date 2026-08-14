# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Resolve ACP ``elicitation/create`` requests — the agent asking the user.

Given the policy, the request and the optional conversation context, return the
response to send back. Deliberately separate from the bridge so policy and
rendering are exercisable without a protocol connection.

The protocol's three response actions map as:

* the user answered -> **accept**
* the user refused, or the mode is one this version does not understand ->
  **decline**
* no HITL channel is available to ask -> **cancel**

Nothing here invents an answer on the user's behalf: with no human reachable the
request is cancelled, never accepted with fabricated content.
"""

import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from acp import schema

from ag2.exceptions import HumanInputNotProvidedError

from .events import ACPElicitation
from .types import ElicitationProperty, ElicitationValue

if TYPE_CHECKING:
    from ag2.context import ConversationContext

    from .config import ElicitationPolicy

logger = logging.getLogger(__name__)

_FORM_MODES = (schema.ElicitationFormSessionMode, schema.ElicitationFormRequestMode)
_URL_MODES = (schema.ElicitationUrlSessionMode, schema.ElicitationUrlRequestMode)

_AFFIRMATIVE = {"y", "yes", "done", "ok", "confirm", "confirmed", "finished"}
_NEGATIVE = {"n", "no", "cancel", "abort", "nope"}

# Typed by the user to refuse a form part-way through. A sentinel rather than a
# bare word because a plain string field may legitimately be answered "no".
REFUSE = "!decline"

# How many times one property may be asked before the request is declined. A
# human who cannot answer a field types `!decline`; a *programmatic* HITL hook
# cannot, and an unbounded re-prompt loop would hang the turn on one that keeps
# answering unusably. Declining after this many tries is the honest way out —
# the field is evidently unanswerable — and never fabricates a value.
MAX_ATTEMPTS = 10


class _RefusedError(Exception):
    """The human refused the request part-way through rendering a form."""


class _UncoercibleError(Exception):
    """The answer cannot be represented as the property's declared type."""


def _mode_name(mode: schema.ElicitationMode) -> str:
    if isinstance(mode, _FORM_MODES):
        return "form"
    if isinstance(mode, _URL_MODES):
        return "url"
    return "other"


def elicitation_event(message: str, mode: schema.ElicitationMode) -> ACPElicitation:
    """The stream event describing this question, whoever ends up answering it."""
    schema_properties = mode.requested_schema.properties if isinstance(mode, _FORM_MODES) else None
    return ACPElicitation(
        message,
        _mode_name(mode),
        url=str(mode.url) if isinstance(mode, _URL_MODES) else None,
        fields=list(schema_properties or ()),
    )


async def resolve_elicitation_response(
    policy: "ElicitationPolicy",
    message: str,
    mode: schema.ElicitationMode,
    context: "ConversationContext | None",
) -> schema.CreateElicitationResponse:
    """Return the ``elicitation/create`` response for one request.

    Args:
        policy: ``"ask"`` | ``"decline"``.
        message: The agent's description of what input it needs.
        mode: The requested mode. One of ACP's form/url mode models, or — once a
            future protocol release adds one — a mode this version has no
            rendering for, which is declined rather than errored.
        context: The conversation context used to ask a human, or ``None``.

    The question is published on the stream before the human is prompted, so an
    observer sees it even when the answer comes from the policy rather than a
    person.
    """
    if context is not None:
        await context.send(elicitation_event(message, mode))

    if policy == "decline":
        return schema.DeclineElicitationResponse(action="decline")

    if not isinstance(mode, _FORM_MODES + _URL_MODES):
        # An `Other`-shaped mode from a later protocol release. Declining teaches
        # the agent to fall back; erroring would fail its turn instead.
        logger.debug("no rendering for ACP elicitation mode %r; declined", type(mode).__name__)
        return schema.DeclineElicitationResponse(action="decline")

    if context is None:
        # Policy says ask, but there is nobody to ask. Cancelling is the only
        # honest answer: a turn must not deadlock on an absent human, and
        # accepting would mean inventing the content.
        return schema.CancelElicitationResponse(action="cancel")

    try:
        if isinstance(mode, _URL_MODES):
            return await _resolve_url(message, mode, context)
        return await _resolve_form(message, mode, context)
    except _RefusedError:
        return schema.DeclineElicitationResponse(action="decline")
    except HumanInputNotProvidedError:
        # The run has a context but no HITL hook behind it — same absent human as
        # above, discovered one prompt later.
        return schema.CancelElicitationResponse(action="cancel")


async def _resolve_url(
    message: str,
    mode: "schema.ElicitationUrlSessionMode | schema.ElicitationUrlRequestMode",
    context: "ConversationContext",
) -> schema.CreateElicitationResponse:
    """Show the message and the URL, then wait for the human to confirm."""
    answer = await context.input(
        f"Agent asks: {message}\nOpen: {mode.url}\nConfirm once you have finished there. Done? (yes/no)"
    )
    if _is_affirmative(answer):
        # url mode requests no fields, so an accept carries no content.
        return schema.AcceptElicitationResponse(action="accept", content=None)
    return schema.DeclineElicitationResponse(action="decline")


async def _resolve_form(
    message: str,
    mode: "schema.ElicitationFormSessionMode | schema.ElicitationFormRequestMode",
    context: "ConversationContext",
) -> schema.CreateElicitationResponse:
    """Render the requested schema as one prompt per property, in schema order.

    A property the human leaves empty takes its default; one that is required
    and has no default is asked again. Answers are coerced to the declared type
    before being returned — the agent asked for a number and must get a number —
    and an uncoercible answer is re-prompted rather than sent through as a
    string.
    """
    properties = mode.requested_schema.properties or {}
    required = set(mode.requested_schema.required or ())

    # Checked up front, not on the way past: a form with a field this version
    # cannot render is declined whole, and declining it on arrival means the human
    # is never asked to fill in fields whose answers would then be thrown away.
    unrenderable = [name for name, prop in properties.items() if _unrenderable(prop)]
    if unrenderable:
        logger.debug("no rendering for ACP elicitation properties %r; declined", unrenderable)
        return schema.DeclineElicitationResponse(action="decline")

    content: dict[str, ElicitationValue] = {}
    for name, prop in properties.items():
        value = await _ask_property(message, name, prop, name in required, context)
        # `is not None`, not truthiness: "" and False and 0 are answers.
        if value is not None:
            content[name] = value

    return schema.AcceptElicitationResponse(action="accept", content=content)


def _unrenderable(prop: ElicitationProperty) -> bool:
    """Whether this property declares a shape there is no prompt for.

    Both cases are additions a later protocol release may make: a property type
    this version has never seen, and a multi-select whose option list is in a
    shape it cannot read. Accepting the latter blind would send the agent values
    it never offered, so it is refused like any other unknown shape.
    """
    if isinstance(prop, schema.ElicitationOtherPropertySchema):
        return True
    if isinstance(prop, schema.ElicitationMultiSelectPropertySchema):
        return not isinstance(prop.items, (schema.StringMultiSelectItems, schema.TitledMultiSelectItems))
    return False


async def _ask_property(
    message: str,
    name: str,
    prop: ElicitationProperty,
    required: bool,
    context: "ConversationContext",
) -> ElicitationValue | None:
    """Prompt for one property until it is answered, defaulted, or skipped.

    Returns the value to send, or ``None`` for an optional property the human left
    empty — omitted from the accepted content rather than filled in with a guess.
    ``None`` carries that on its own: a coerced answer is never ``None``, and a
    default is only taken when the property declares one.

    Raises:
        _RefusedError: The human refused, or gave ``MAX_ATTEMPTS`` answers none of
            which the property could take.
    """
    prompt = _prompt_for(message, name, prop)
    for _ in range(MAX_ATTEMPTS):
        answer = (await context.input(prompt)).strip()
        if answer == REFUSE:
            raise _RefusedError
        if not answer:
            default: ElicitationValue | None = getattr(prop, "default", None)
            if default is not None:
                return default
            if not required:
                return None
            continue  # required, no default -> ask again
        try:
            return _coerce(answer, prop)
        except _UncoercibleError:
            continue
    logger.debug("property %r went unanswered after %d attempts; declined", name, MAX_ATTEMPTS)
    raise _RefusedError


def _prompt_for(message: str, name: str, prop: ElicitationProperty) -> str:
    """One property's prompt: what it is, what it accepts, and what it defaults to."""
    title: str | None = getattr(prop, "title", None)
    lines = [f"Agent asks: {message}", f"{title or name} ({_type_label(prop)}):"]
    description: str | None = getattr(prop, "description", None)
    if description:
        lines.append(description)
    allowed = _allowed_values(prop)
    if allowed is not None:
        lines.append(f"Allowed values: {', '.join(allowed)}")
    bounds = _bounds_note(prop)
    if bounds is not None:
        lines.append(bounds)
    default: ElicitationValue | None = getattr(prop, "default", None)
    if default is not None:
        lines.append(f"Default (press enter to accept): {_render_default(default)}")
    lines.append(f"Answer, or {REFUSE} to refuse:")
    return "\n".join(lines)


def _type_label(prop: ElicitationProperty) -> str:
    if isinstance(prop, schema.ElicitationMultiSelectPropertySchema):
        return "select any, comma-separated"
    if isinstance(prop, schema.ElicitationBooleanPropertySchema):
        return "yes/no"
    return str(getattr(prop, "type", "string"))


def _render_default(default: ElicitationValue) -> str:
    if isinstance(default, list):
        return ", ".join(str(item) for item in default)
    if isinstance(default, bool):
        return "yes" if default else "no"
    return str(default)


def _allowed_values(prop: ElicitationProperty) -> list[str] | None:
    """The values this property declares, or ``None`` when it declares none."""
    if isinstance(prop, schema.ElicitationStringPropertySchema):
        if prop.enum:
            return list(prop.enum)
        if prop.one_of:
            return [option.const for option in prop.one_of]
        return None
    if isinstance(prop, schema.ElicitationMultiSelectPropertySchema):
        items = prop.items
        if isinstance(items, schema.StringMultiSelectItems):
            return list(items.enum)
        if isinstance(items, schema.TitledMultiSelectItems):
            return [option.const for option in items.any_of]
    return None


def _bounds_note(prop: ElicitationProperty) -> str | None:
    """The numeric bounds this property declares, phrased for the human.

    Shown because they are *enforced*: a re-prompt after breaking a limit the
    prompt never mentioned would look like the answer was simply ignored.
    ``pattern`` and ``format`` are deliberately not enforced — a regex the user
    cannot see is not something to re-prompt against, and a string reaches the
    agent either way for it to judge.
    """
    if isinstance(prop, (schema.ElicitationNumberPropertySchema, schema.ElicitationIntegerPropertySchema)):
        return _range_note("Must be", prop.minimum, prop.maximum)
    if isinstance(prop, schema.ElicitationMultiSelectPropertySchema):
        return _range_note("Pick", prop.min_items, prop.max_items, suffix=" values")
    if isinstance(prop, schema.ElicitationStringPropertySchema):
        return _range_note("Length", prop.min_length, prop.max_length, suffix=" characters")
    return None


def _range_note(label: str, low: float | None, high: float | None, suffix: str = "") -> str | None:
    if low is not None and high is not None:
        return f"{label} between {low} and {high}{suffix}"
    if low is not None:
        return f"{label} at least {low}{suffix}"
    if high is not None:
        return f"{label} at most {high}{suffix}"
    return None


def _within(value: float, low: float | None, high: float | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def _coerce(answer: str, prop: ElicitationProperty) -> ElicitationValue:
    """Coerce one text answer to the value the property declared it wants.

    Raises:
        _UncoercibleError: The answer is not a value this property can take —
            the wrong type, or outside the values or bounds it declares. The
            caller re-prompts rather than sending a string where a number was
            asked for, or a number the agent already said it cannot use.
    """
    if isinstance(prop, schema.ElicitationIntegerPropertySchema):
        return _coerce_number(answer, int, prop.minimum, prop.maximum)
    if isinstance(prop, schema.ElicitationNumberPropertySchema):
        return _coerce_number(answer, float, prop.minimum, prop.maximum)
    if isinstance(prop, schema.ElicitationBooleanPropertySchema):
        return _coerce_bool(answer)
    if isinstance(prop, schema.ElicitationMultiSelectPropertySchema):
        return _coerce_multi_select(answer, prop)
    return _coerce_string(answer, prop)


def _coerce_number(
    answer: str,
    parse: Callable[[str], int] | Callable[[str], float],
    minimum: float | None,
    maximum: float | None,
) -> int | float:
    try:
        value = parse(answer)
    except ValueError:
        raise _UncoercibleError from None
    # "nan"/"inf" parse as floats but have no JSON form: they would cross the wire
    # as `null`, handing the agent a null where it asked for a number.
    if not math.isfinite(value) or not _within(value, minimum, maximum):
        raise _UncoercibleError
    return value


def _coerce_bool(answer: str) -> bool:
    lowered = answer.lower()
    if lowered in _AFFIRMATIVE or lowered in {"true", "1", "on"}:
        return True
    if lowered in _NEGATIVE or lowered in {"false", "0", "off"}:
        return False
    raise _UncoercibleError


def _coerce_string(answer: str, prop: ElicitationProperty) -> str:
    allowed = _allowed_values(prop)
    # A value outside the declared set is as wrong as a word where a number was
    # asked for: the agent branches on those values.
    if allowed is not None and answer not in allowed:
        raise _UncoercibleError
    if isinstance(prop, schema.ElicitationStringPropertySchema) and not _within(
        len(answer), prop.min_length, prop.max_length
    ):
        raise _UncoercibleError
    return answer


def _coerce_multi_select(answer: str, prop: schema.ElicitationMultiSelectPropertySchema) -> list[str]:
    selected = [part.strip() for part in answer.split(",") if part.strip()]
    if not selected:
        raise _UncoercibleError
    allowed = _allowed_values(prop)
    if allowed is not None and any(item not in allowed for item in selected):
        raise _UncoercibleError
    if not _within(len(selected), prop.min_items, prop.max_items):
        raise _UncoercibleError
    return selected


def _is_affirmative(answer: str) -> bool:
    return answer.strip().lower() in _AFFIRMATIVE
