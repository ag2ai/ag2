# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Iterable

from autogen.beta.context import StreamId
from autogen.beta.events import BaseEvent, UnknownEvent
from autogen.beta.events._serialization import import_event_class, qualified_name

from .base import KnowledgeStore


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
        path = f"/log/{stream_id}.jsonl"
        lines = self._serialize_events(events)
        await self._store.write(path, "\n".join(lines))

    async def persist_dropped(self, stream_id: StreamId, events: Iterable[BaseEvent]) -> None:
        """Write compaction-dropped events to /log/{stream_id}.dropped-{n}.jsonl.

        Discovers existing segments in the store to avoid overwriting.
        """
        prefix = f"{stream_id}.dropped-"
        entries = await self._store.list("/log/")
        existing = [e for e in entries if e.startswith(prefix) and e.endswith(".jsonl")]
        n = len(existing) + 1
        path = f"/log/{stream_id}.dropped-{n}.jsonl"
        lines = self._serialize_events(events)
        await self._store.write(path, "\n".join(lines))

    async def load(self, stream_id: StreamId) -> list[BaseEvent]:
        """Load events from WAL files: all dropped segments in order, then final.

        Returns typed BaseEvent instances. Unknown types become UnknownEvent.
        """
        all_events: list[BaseEvent] = []

        entries = await self._store.list("/log/")
        prefix = f"{stream_id}.dropped-"
        segments = sorted(
            [e for e in entries if e.startswith(prefix) and e.endswith(".jsonl")],
            key=lambda e: int(e[len(prefix) : -len(".jsonl")]),
        )
        for segment in segments:
            events = await self._load_file(f"/log/{segment}")
            all_events.extend(events)

        final = await self._load_file(f"/log/{stream_id}.jsonl")
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
