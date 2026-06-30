# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical A2UI wire serialization: :func:`to_jsonl` renders a list of
server→client messages as JSON Lines (one JSON message per line). Pure, with no
transport dependencies; transport adapters build on top of it.
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
