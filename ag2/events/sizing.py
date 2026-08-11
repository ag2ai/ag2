# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Content-size estimation and prompt rendering for history management.

Sizes and renders an event from its full content (never the truncated repr):
text verbatim, non-text parts by a flat per-modality token budget.
"""

from collections.abc import Iterator

from .base import BaseEvent, is_conversational
from .input_events import BinaryInput, BinaryType, DataInput, FileIdInput, ModelRequest, TextInput, UrlInput
from .tool_events import ToolResultsEvent
from .types import ModelResponse

# One flat per-modality token budget for all providers. A rough heuristic: real
# image cost tracks pixel dimensions and differs per provider, so it isn't exact.
_MODALITY_TOKENS = {
    "image": 1000,
    "audio": 1000,
    "video": 2000,
    "document": 2000,
    "binary": 1000,
}

# (kind, payload) where kind is "text" (payload is the text) or "media"
# (payload is a modality key into _MODALITY_TOKENS).
_Piece = tuple[str, str]

_LABELS = ((ModelRequest, "User: "), (ModelResponse, "Assistant: "), (ToolResultsEvent, "Tool: "))


def _modality(kind: BinaryType | None, media_type: str | None) -> str:
    if kind is not None and kind != BinaryType.BINARY:
        return kind.value
    if media_type:
        return media_type.split("/", 1)[0]
    return "binary"


def _part_pieces(parts: list) -> Iterator[_Piece]:
    for p in parts:
        if isinstance(p, TextInput):
            yield ("text", p.content)
        elif isinstance(p, DataInput):
            yield ("text", str(p.data))
        elif isinstance(p, BinaryInput):
            yield ("media", _modality(p.kind, p.media_type))
        elif isinstance(p, UrlInput):
            yield ("media", _modality(p.kind, None))
        elif isinstance(p, FileIdInput):
            yield ("media", "binary")
        else:
            yield ("text", str(p))


def _content_pieces(event: BaseEvent) -> Iterator[_Piece]:
    if isinstance(event, ModelRequest):
        yield from _part_pieces(event.parts)
    elif isinstance(event, ToolResultsEvent):
        for r in event.results:
            yield from _part_pieces(r.result.parts)
    elif isinstance(event, ModelResponse):
        if event.message and event.message.content:
            yield ("text", event.message.content)
        for call in event.tool_calls.calls:
            yield ("text", f"{call.name}({call.arguments})")
        for _ in event.files:
            yield ("media", "binary")
    else:
        # Duck-typed content/summary covers HumanMessage, CompactionSummary, etc.;
        # str() is the last resort for any other type.
        for attr in ("content", "summary"):
            value = getattr(event, attr, None)
            if isinstance(value, str):
                yield ("text", value)
                return
        yield ("text", str(event))


def estimated_tokens(event: BaseEvent, chars_per_token: int = 4) -> int:
    """Rough token size of an event's content, never truncated.

    Text counts as ``len // chars_per_token``; each non-text part counts as a
    flat per-modality budget. A size heuristic, not an exact tokenizer.
    """
    # Telemetry never reaches the model, so it occupies no context budget.
    if not is_conversational(event):
        return 0
    total = 0
    for kind, payload in _content_pieces(event):
        if kind == "text":
            total += len(payload) // chars_per_token
        else:
            total += _MODALITY_TOKENS.get(payload, _MODALITY_TOKENS["binary"])
    return total


def render_for_prompt(event: BaseEvent) -> str:
    """Full, untruncated text of an event for a summarizer/memory prompt.

    Text parts are rendered in full; non-text parts become ``[modality]``
    placeholders. Prefixed with a role label where one applies. Telemetry
    (non-conversational events) renders empty so it never enters a prompt.
    """
    if not is_conversational(event):
        return ""
    pieces = [payload if kind == "text" else f"[{payload}]" for kind, payload in _content_pieces(event)]
    body = " ".join(p for p in pieces if p)
    if not body:
        return str(event)
    label = next((lbl for cls, lbl in _LABELS if isinstance(event, cls)), "")
    return f"{label}{body}"
