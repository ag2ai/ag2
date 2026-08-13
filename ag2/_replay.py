# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Replay invariants for a trimmed span of history.

Every reducer of history — the assembly policies, the limiter middleware, and the
compaction strategies — cuts the event list at an index and replays
``events[cut:]`` to the provider. That tail has to stand on its own, and four
retained events cannot:

| retained event             | required companion             | if missing                      |
|----------------------------|--------------------------------|---------------------------------|
| tool result                | the **call** it answers        | orphan ``function_call_output`` |
| builtin (server-side) call | its **reasoning** item         | orphan ``web_search_call``      |
| local call event           | the **response** announcing it | maps to nothing at all          |
| response with tool calls   | its **provider turn item**     | tool calls vanish, silently     |

The remedy differs by caller because what they own differs: a policy or limiter
persists nothing, so it drops the offending event (:func:`replayable_span`);
compaction persists everything it drops, so it moves the cut (:func:`snap`) —
filtering there would leave an event neither retained nor persisted.
"""

from collections.abc import Sequence

from ag2.events import (
    BaseEvent,
    BuiltinToolCallEvent,
    ModelRequest,
    ModelResponse,
    ProviderReplay,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultEvent,
    ToolResultsEvent,
)


def _answerable_ids(events: Sequence[BaseEvent]) -> set[str]:
    """Ids of calls a span can answer — local calls and builtin ones alike.

    A builtin call arrives as its own event and never appears in
    ``ModelResponse.tool_calls``, so a check that reads only the response misses
    it and concludes its result is orphaned no matter what is retained.
    """
    ids: set[str] = set()
    for event in events:
        if isinstance(event, ModelResponse) and event.tool_calls:
            ids.update(call.id for call in event.tool_calls.calls)
        elif isinstance(event, BuiltinToolCallEvent):
            ids.add(event.id)
    return ids


def _required_ids(event: BaseEvent) -> set[str]:
    """Call ids this event is a result for; empty for everything else."""
    if isinstance(event, ToolResultsEvent):
        return {result.parent_id for result in event.results if result.parent_id}
    if isinstance(event, ToolResultEvent) and event.parent_id:
        return {event.parent_id}
    return set()


def _is_anchor(event: BaseEvent) -> bool:
    """True for an item the builtin calls of its response are paired with.

    ``ProviderReplay`` marks the requirement and the event's own
    ``__replay_role__`` names which half of it applies, so neither durability nor
    the event's base classes decide this. Durability answers a different question
    (storage, not replay), and reading the role off ``ModelReasoning`` would
    misfile any provider whose turn carrier happened to subclass it.
    """
    return isinstance(event, ProviderReplay) and event.__replay_role__ == "anchor"


def _is_turn_item(event: BaseEvent) -> bool:
    """True for provider-native state standing in for a whole assistant turn.

    The other ``ProviderReplay`` role: some providers hand back an object that is
    the only way to reconstruct the turn they just produced. ``XAIAssistantEvent``
    carries the response proto, and xai-sdk offers no way to build an assistant
    message with ``tool_calls`` from primitives. Anchors are excluded — they cover
    a single builtin call, which is a narrower relationship than a whole turn.
    """
    return isinstance(event, ProviderReplay) and event.__replay_role__ == "turn"


def _prune(events: Sequence[BaseEvent], cut: int) -> list[BaseEvent]:
    """``events[cut:]`` with every event the cut orphaned removed.

    ``anchor`` is the nearest preceding durable reasoning item, tracked across the
    whole list rather than read off the events adjacent to the cut. That makes the
    answer independent of whatever else the provider interleaved into the response,
    and silent for models that emit no reasoning at all — a builtin call that
    never had an anchor cannot lose one.

    It is scoped to one response, though: a reasoning item anchors only the builtin
    calls of the response that emitted it, and ``ModelRequest`` / ``ModelResponse``
    close that response. Without the reset a stale anchor from an earlier response
    would condemn a later builtin call that never needed one — and a history can
    genuinely mix the two, since the same model emits a reasoning item only when
    asked for a summary (see ``ag2.config.openai.openai_responses_client``).
    """
    anchor: int | None = None
    turn_item: int | None = None
    kept: list[BaseEvent] = []
    for index, event in enumerate(events):
        orphaned_turn = False
        if _is_anchor(event):
            anchor = index
        elif _is_turn_item(event):
            turn_item = index
        elif isinstance(event, ModelRequest):
            anchor = None
            turn_item = None
        elif isinstance(event, ModelResponse):
            anchor = None
            # The response consumes the turn item emitted just before it. Only
            # tool calls are lost without it — the text is on the response
            # itself — so a plain answer survives the item being cut away.
            orphaned_turn = turn_item is not None and turn_item < cut and bool(event.tool_calls.calls)
            turn_item = None
        if index < cut:
            continue
        if isinstance(event, BuiltinToolCallEvent) and anchor is not None and anchor < cut:
            continue
        if orphaned_turn:
            continue
        kept.append(event)

    answerable = _answerable_ids(kept)
    return [event for event in kept if _required_ids(event) <= answerable and _announced_ids(event) <= answerable]


def _announced_ids(event: BaseEvent) -> set[str]:
    """Call ids this event merely announces, carrying no replayable content.

    A local call is announced twice: once inside the ``ModelResponse`` that
    requested it, which is what a provider replays, and once as a standalone
    event (or a container of them) for observers. Without the response those
    standalone events map to no input item at all, so a span reduced to them is
    as unsendable as an empty one — the mirror of a builtin call with no anchor.
    Builtin calls are excluded: they carry their own provider item.
    """
    if isinstance(event, ToolCallsEvent):
        return {call.id for call in event.calls}
    if isinstance(event, ToolCallEvent) and not isinstance(event, BuiltinToolCallEvent):
        return {event.id}
    return set()


def replayable_span(events: Sequence[BaseEvent], cut: int) -> list[BaseEvent]:
    """Return ``events[cut:]`` reduced to what a provider will accept on its own.

    Drops orphans rather than moving the cut, so a window loses only the events
    that cannot be replayed. Retreats the cut only to avoid the one outcome no
    provider accepts either — a request with nothing in it — which is what a
    window narrower than the turn it lands in reduces to once its orphans are
    gone. Overshooting a window to reach the turn is recoverable; sending
    nothing is not.
    """
    kept = _prune(events, cut)
    while not kept and cut > 0:
        cut -= 1
        kept = _prune(events, cut)
    return kept


def is_replayable(events: Sequence[BaseEvent], cut: int) -> bool:
    """True when ``events[cut:]`` needs nothing the cut dropped."""
    return len(_prune(events, cut)) == len(events) - cut


def snap(events: Sequence[BaseEvent], cut: int) -> int:
    """Advance ``cut`` to the next index whose retained span is replayable.

    Advancing only ever drops more events, so a caller's budget still holds
    afterwards, and the events skipped over are the caller's to persist. A cut of
    0 trims nothing and is returned untouched: a caller that decided to keep
    everything must not be turned into one that drops events.
    """
    if cut <= 0:
        return 0
    while cut < len(events) and not is_replayable(events, cut):
        cut += 1
    return cut
