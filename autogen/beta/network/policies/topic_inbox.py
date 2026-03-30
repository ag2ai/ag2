# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""TopicInboxPolicy — injects unread topic messages into context."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from autogen.beta.context import Context
from autogen.beta.events import BaseEvent

if TYPE_CHECKING:
    from autogen.beta.config import ModelConfig

    from ..events import TopicMessage
    from ..hub import Hub


class TopicOverflow(str, Enum):
    """How to handle topic message backlogs."""

    NEWEST = "newest"  # Keep the N newest messages. Default.
    OLDEST = "oldest"  # Keep the N oldest messages.
    SUMMARY = "summary"  # Summarize the backlog into one message.


class TopicInboxPolicy:
    """Injects unread topic messages into the actor's context.

    Reads new messages from all subscribed topics via the Hub.
    Injects them as system prompt context. Advances the read cursor
    so messages are not re-injected.
    """

    name = "topic_inbox"

    def __init__(
        self,
        hub: Hub,
        actor_name: str,
        max_messages: int = 50,
        overflow: TopicOverflow = TopicOverflow.NEWEST,
        summary_config: ModelConfig | None = None,
    ) -> None:
        self._hub = hub
        self._actor = actor_name
        self._max = max_messages
        self._overflow = overflow
        self._summary_config = summary_config

    async def apply(
        self,
        prompts: list[str],
        events: list[BaseEvent],
        context: Context,
    ) -> tuple[list[str], list[BaseEvent]]:
        subscriptions = self._hub.subscriptions_for(self._actor)

        # Peek without advancing cursors so overflow strategies
        # can decide which messages to consume.
        per_topic: list[tuple[str, list[TopicMessage]]] = []
        all_messages: list[TopicMessage] = []
        for topic in subscriptions:
            messages = await self._hub.peek_topic(self._actor, topic)
            per_topic.append((topic, messages))
            all_messages.extend(messages)

        if not all_messages:
            return prompts, events

        overflow_note = ""

        # Apply overflow strategy
        if len(all_messages) > self._max:
            if self._overflow == TopicOverflow.NEWEST:
                dropped = len(all_messages) - self._max
                all_messages = all_messages[-self._max:]
                overflow_note = f"({dropped} older messages omitted)"
                # Advance all cursors — older messages are intentionally skipped
                for topic, messages in per_topic:
                    await self._hub.advance_topic(self._actor, topic, len(messages))
            elif self._overflow == TopicOverflow.OLDEST:
                kept = all_messages[:self._max]
                dropped = len(all_messages) - self._max
                all_messages = kept
                overflow_note = f"({dropped} newer messages deferred)"
                # Advance only for the messages we consumed
                consumed = self._max
                for topic, messages in per_topic:
                    take = min(len(messages), consumed)
                    if take > 0:
                        await self._hub.advance_topic(self._actor, topic, take)
                    consumed -= take
                    if consumed <= 0:
                        break
            elif self._overflow == TopicOverflow.SUMMARY:
                summary = await self._summarize(all_messages)
                # Advance all cursors — everything was summarized
                for topic, messages in per_topic:
                    await self._hub.advance_topic(self._actor, topic, len(messages))
                prompts = prompts + [f"## Network Messages (summarized)\n\n{summary}"]
                return prompts, events
        else:
            # No overflow — advance all cursors
            for topic, messages in per_topic:
                await self._hub.advance_topic(self._actor, topic, len(messages))

        lines = ["## Network Messages\n"]
        if overflow_note:
            lines.append(f"*{overflow_note}*\n")
        for msg in all_messages:
            lines.append(f"- **[{msg.topic}]** {msg.sender}: {msg.message}")
        prompts = prompts + ["\n".join(lines)]

        return prompts, events

    async def _summarize(self, messages: list[TopicMessage]) -> str:
        """Summarize a backlog of topic messages via LLM."""
        if not self._summary_config:
            # Fallback: just list recent messages
            return "\n".join(f"[{m.topic}] {m.sender}: {m.message}" for m in messages[-self._max:])

        from autogen.beta.context import Context as Ctx
        from autogen.beta.events import ModelRequest
        from autogen.beta.stream import MemoryStream

        client = self._summary_config.create()
        formatted = "\n".join(f"[{m.topic}] {m.sender}: {m.message}" for m in messages)
        response = await client(
            [ModelRequest(content=f"Summarize these {len(messages)} network messages concisely:\n\n{formatted}")],
            Ctx(MemoryStream()),
            tools=[],
            response_schema=None,
        )
        return response.content or ""
