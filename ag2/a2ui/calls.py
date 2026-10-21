# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded memory of the ``callRendererFunction`` calls a runtime has issued, so a
later ``rendererFunctionResponse`` can be reported to the agent under the
function's name. Per-process and best-effort: a miss (restart, another worker)
only drops the name from the prompt.
"""

from collections import OrderedDict
from typing import cast

from ._types import CallRendererFunctionMessage, ServerToClientMessage

_DEFAULT_CAPACITY = 256


class RendererCallLedger:
    """LRU map of ``functionCallId`` → function name for issued ``callRendererFunction`` calls."""

    __slots__ = ("_capacity", "_names")

    def __init__(self, capacity: int = _DEFAULT_CAPACITY) -> None:
        self._capacity = capacity
        self._names: OrderedDict[str, str] = OrderedDict()

    def record(self, message: ServerToClientMessage) -> None:
        """Remember the call carried by *message*; a no-op for any other message."""
        if "callRendererFunction" not in message:
            return
        content = cast(CallRendererFunctionMessage, message)["callRendererFunction"]
        call = content.get("callFunction")
        function_call_id = content.get("functionCallId")
        name = call.get("@call") if call else None
        if not function_call_id or not name:
            return
        self._names[function_call_id] = name
        self._names.move_to_end(function_call_id)
        while len(self._names) > self._capacity:
            self._names.popitem(last=False)

    def name_for(self, function_call_id: str) -> str | None:
        """The function name issued under *function_call_id*, if still remembered."""
        return self._names.get(function_call_id)


__all__ = ("RendererCallLedger",)
