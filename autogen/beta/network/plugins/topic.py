# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TopicPlugin — pub/sub messaging between actors.

Extracted from Hub to keep Hub focused on registry + delegation.
Topics provide cursor-based message streams that actors can publish
to and subscribe from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..events import TopicMessage, TopicSubscription, TopicUnsubscription
from ..topology import BasePlugin

if TYPE_CHECKING:
    from ..hub import Hub


class TopicPlugin(BasePlugin):
    """Pub/sub messaging between actors.

    Manages topic state: message buffers, subscriptions, and cursors.
    Installed on a Hub via the plugin system.

    Example::

        hub = Hub(plugins=[TopicPlugin()])
    """

    name = "topics"

    def __init__(self) -> None:
        super().__init__()
        self._topics: dict[str, list[TopicMessage]] = {}
        self._subscriptions: dict[str, set[str]] = {}  # topic -> actor names
        self._cursors: dict[tuple[str, str], int] = {}  # (actor, topic) -> last-read index
        self._hub: Hub | None = None

    def install(self, hub: Hub) -> None:
        self._hub = hub

    def uninstall(self, hub: Hub) -> None:
        self._hub = None
        self._topics.clear()
        self._subscriptions.clear()
        self._cursors.clear()

    async def publish(self, sender: str, topic: str, message: str, data: dict | None = None) -> None:
        """Publish a message to a topic."""
        msg = TopicMessage(topic=topic, sender=sender, message=message, data=data or {})
        self._topics.setdefault(topic, []).append(msg)
        if self._hub:
            await self._hub._emit(msg)

    async def subscribe_topic(self, actor_name: str, topic: str) -> None:
        """Subscribe an actor to a topic. Cursor starts at current end (no replay)."""
        self._subscriptions.setdefault(topic, set()).add(actor_name)
        self._cursors[(actor_name, topic)] = len(self._topics.get(topic, []))
        if self._hub:
            await self._hub._emit(TopicSubscription(actor=actor_name, topic=topic))

    async def unsubscribe_topic(self, actor_name: str, topic: str) -> None:
        """Unsubscribe an actor from a topic."""
        self._subscriptions.get(topic, set()).discard(actor_name)
        self._cursors.pop((actor_name, topic), None)
        if self._hub:
            await self._hub._emit(TopicUnsubscription(actor=actor_name, topic=topic))

    async def read_topic(self, actor_name: str, topic: str) -> list[TopicMessage]:
        """Read new messages from a topic since last read. Advances cursor."""
        cursor = self._cursors.get((actor_name, topic), 0)
        messages = self._topics.get(topic, [])
        new_messages = messages[cursor:]
        self._cursors[(actor_name, topic)] = len(messages)
        return new_messages

    async def peek_topic(self, actor_name: str, topic: str) -> list[TopicMessage]:
        """Read new messages without advancing cursor."""
        cursor = self._cursors.get((actor_name, topic), 0)
        return self._topics.get(topic, [])[cursor:]

    async def advance_topic(self, actor_name: str, topic: str, count: int) -> None:
        """Advance cursor by count messages."""
        key = (actor_name, topic)
        current = self._cursors.get(key, 0)
        max_pos = len(self._topics.get(topic, []))
        self._cursors[key] = min(current + count, max_pos)

    async def list_topics(self) -> list[str]:
        """List all active topics."""
        return list(self._topics.keys())

    def subscriptions_for(self, actor_name: str) -> list[str]:
        """List topics an actor is subscribed to."""
        return [topic for topic, actors in self._subscriptions.items() if actor_name in actors]

    def cleanup_actor(self, actor_name: str) -> None:
        """Remove all subscriptions and cursors for an actor."""
        for subscribers in self._subscriptions.values():
            subscribers.discard(actor_name)
        self._cursors = {k: v for k, v in self._cursors.items() if k[0] != actor_name}
