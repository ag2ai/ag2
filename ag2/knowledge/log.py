# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable

from ag2.context import StreamId
from ag2.events import BaseEvent, UnknownEvent
from ag2.events._serialization import import_event_class, qualified_name

from .base import KnowledgeStore
from .constants import LOG_PREFIX


def _dropped_segments(entries: Iterable[str], prefix: str) -> list[tuple[int, str]]:
    """``(n, entry)`` for each ``{prefix}{n}.jsonl`` entry, in segment order; non-integer ``n`` is ignored."""
    segments: list[tuple[int, str]] = []
    for entry in entries:
        if entry.startswith(prefix) and entry.endswith(".jsonl"):
            suffix = entry[len(prefix) : -len(".jsonl")]
            if suffix.isascii() and suffix.isdigit():
                segments.append((int(suffix), entry))
    return sorted(segments)


class EventLogWriter:
    """Persists stream events to the knowledge store as WAL entries.

    Each event is serialized as a JSON line with a type tag for
    deserialization. Uses append-segmented writes: dropped events from
    compaction go to numbered segment files, final events go to the
    main log file.
    """

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    async def persist(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write final events to /log/{stream_id}.jsonl."""
        path = f"{LOG_PREFIX}{stream_id}.jsonl"
        lines = self._serialize_events(events)
        await self._store.write(path, "\n".join(lines))

    async def persist_dropped(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write compaction-dropped events to /log/{stream_id}.dropped-{n}.jsonl.

        Concurrent writers may share a segment; atomic appends preserve both batches.
        """
        prefix = f"{stream_id}.dropped-"
        entries = await self._store.list(LOG_PREFIX)
        # Number after the highest existing segment, not the count: once an older
        # segment is gone, the count would reuse (and overwrite) a live one.
        segments = _dropped_segments(entries, prefix)
        n = (segments[-1][0] if segments else 0) + 1
        path = f"{LOG_PREFIX}{stream_id}.dropped-{n}.jsonl"
        lines = self._serialize_events(events)
        await self._store.append(path, "\n".join(lines) + "\n")

    async def load(self, stream_id: StreamId) -> list[BaseEvent]:
        """Load events from WAL files: all dropped segments in order, then final.

        Returns typed BaseEvent instances. Unknown types become UnknownEvent.
        """
        all_events: list[BaseEvent] = []

        entries = await self._store.list(LOG_PREFIX)
        prefix = f"{stream_id}.dropped-"
        for _, segment in _dropped_segments(entries, prefix):
            events = await self._load_file(f"{LOG_PREFIX}{segment}")
            all_events.extend(events)

        final = await self._load_file(f"{LOG_PREFIX}{stream_id}.jsonl")
        all_events.extend(final)

        return all_events

    def _serialize_events(self, events: Iterable[BaseEvent]) -> list[str]:
        lines: list[str] = []
        for event in events:
            record = {
                "type": qualified_name(event),
                "data": event.to_dict(),
            }
            lines.append(json.dumps(record, default=str))
        return lines

    async def _load_file(self, path: str) -> list[BaseEvent]:
        content = await self._store.read(path)
        if not content:
            return []

        events: list[BaseEvent] = []
        for line in content.strip().split("\n"):
            if not line:
                continue
            record = json.loads(line)
            event_type = record["type"]
            event_data = record["data"]

            cls: type[BaseEvent] | None = import_event_class(event_type)

            if cls is not None:
                try:
                    events.append(cls.from_dict(event_data))
                except Exception:
                    events.append(UnknownEvent(type_name=event_type, data=event_data))
            else:
                events.append(UnknownEvent(type_name=event_type, data=event_data))

        return events
