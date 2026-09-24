# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``observer()``'s return type, for every way of passing its callback.

Checked by ``test/typing/test_fixtures.py``; not imported by the suite. ``reveal_type``
lines are the assertions — mypy reports them as notes.
"""

from ag2.events import BaseEvent, ToolCallEvent
from ag2.observers import observer


def on_event(event: BaseEvent) -> None: ...


# Without a condition there is nothing to filter on, so a bare `SimpleObserver` comes back.
reveal_type(observer(None, on_event))  # N: Revealed type is "ag2.observers.observer.SimpleObserver"
reveal_type(observer(callback=on_event))  # N: Revealed type is "ag2.observers.observer.SimpleObserver"
reveal_type(observer()(on_event))  # N: Revealed type is "ag2.observers.observer.SimpleObserver"

# With one, a `StreamObserver`.
reveal_type(observer(ToolCallEvent, on_event))  # N: Revealed type is "ag2.observers.observer.StreamObserver"
reveal_type(observer(ToolCallEvent, callback=on_event))  # N: Revealed type is "ag2.observers.observer.StreamObserver"
reveal_type(observer(ToolCallEvent)(on_event))  # N: Revealed type is "ag2.observers.observer.StreamObserver"
