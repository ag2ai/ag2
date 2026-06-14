# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical A2UI wire serialization.

The A2UI core represents a server→client exchange as a list of typed
messages (``list[ServerToClientMessage]``). The canonical A2UI wire format
is **JSON Lines (JSONL)**: one JSON message per line, streaming-friendly and
easy for LLMs to generate incrementally.

:func:`to_jsonl` is a pure function with no transport dependencies — transport
adapters (A2A DataParts, AG-UI events, REST/SSE) build on top of it. The
agent's delimiter-based parser (``text + delimiter + JSON array``) is an AG2
convenience for extracting both conversational text and UI from a single LLM
response; it is *not* part of the A2UI wire format.
"""

import json
from collections.abc import Sequence

from ._types import ServerToClientMessage


def to_jsonl(messages: Sequence[ServerToClientMessage]) -> str:
    """Serialize A2UI messages to canonical JSONL (one message per line).

    Args:
        messages: The A2UI server→client messages to serialize.

    Returns:
        A JSONL string with one compact JSON object per line and no trailing
        newline. An empty sequence yields an empty string.
    """
    return "\n".join(json.dumps(m, separators=(",", ":")) for m in messages)


__all__ = ("to_jsonl",)
