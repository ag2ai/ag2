# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""LLM-facing tools attached by ``NetworkPlugin``.

Two flat tools cover the hot path; four grouped action-dispatch tools
cover discovery and lifecycle:

Flat:
* ``say`` — post into a channel.
* ``delegate`` — one-shot consult.

Grouped:
* ``peers``    — find / describe peers.
* ``channels`` — list / open / info / close.
* ``tasks``    — progress / complete (active) + list / status / wait.
* ``context``  — search / quote past content.
"""

from .channels import make_channels_tool
from .context import make_context_tool
from .delegate import make_delegate_tool
from .peers import make_peers_tool
from .say import make_say_tool
from .tasks import make_tasks_tool

__all__ = (
    "make_channels_tool",
    "make_context_tool",
    "make_delegate_tool",
    "make_peers_tool",
    "make_say_tool",
    "make_tasks_tool",
)
