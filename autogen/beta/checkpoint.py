# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoints — save and restore agent conversation state.

A ``Checkpoint`` captures the event history and context variables after a
turn, letting you resume a conversation in a new process or at a later time.
The agent's code (tools, middleware, prompts, LLM config) is **not** saved —
you reconstruct the Agent normally and feed it the checkpoint.

Usage::

    # Save a checkpoint after a turn
    reply = await agent.ask("plan my week")
    cp = await Checkpoint.from_reply(reply)
    cp.save("planner.json")

    # Resume in a new process
    cp = Checkpoint.load("planner.json")
    stream = await MemoryStream.from_checkpoint(cp)
    reply = await agent.ask("add a gym session on Wednesday", stream=stream, variables=cp.variables)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .events._serialization import deserialize_value, qualified_name
from .events.base import BaseEvent

if TYPE_CHECKING:
    from .agent import AgentReply

_VERSION = "1"


class Checkpoint:
    """Snapshot of an agent's conversation state (event history + context variables).

    Create with ``await Checkpoint.from_reply(reply)`` after any ``agent.ask`` call.
    Restore with ``MemoryStream.from_checkpoint(cp)`` and pass the resulting stream
    (plus ``variables=cp.variables``) to the next ``agent.ask`` call.

    Only runtime state is captured — the agent's tools, prompts, middleware, and
    LLM config are not serialized. Reconstruct the Agent normally when resuming.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        stream_id: UUID,
        events: list[dict[str, Any]],
        variables: dict[str, Any],
        created_at: datetime | None = None,
        version: str = _VERSION,
    ) -> None:
        self.agent_name = agent_name
        self.stream_id = stream_id
        self.events = events
        self.variables = variables
        self.created_at = created_at or datetime.now(tz=timezone.utc)
        self.version = version

    @classmethod
    async def from_reply(cls, reply: "AgentReply[Any, Any]") -> "Checkpoint":
        """Capture conversation state from a completed turn's reply.

        Args:
            reply: The ``AgentReply`` returned by ``agent.ask()``.

        Returns:
            A ``Checkpoint`` containing the stream's event history and the
            context variables at the end of that turn.
        """
        raw_events = list(await reply.history.get_events())
        serialized = [{"__event__": qualified_name(e), **e.to_dict()} for e in raw_events]
        return cls(
            agent_name=reply.agent_name,
            stream_id=reply.context.stream.id,
            events=serialized,
            variables=dict(reply.context.variables),
        )

    def restore_events(self) -> list[BaseEvent]:
        """Deserialize the stored events back to ``BaseEvent`` instances."""
        return [deserialize_value(e) for e in self.events]  # type: ignore[return-value]

    def save(self, path: str | Path) -> None:
        """Write this checkpoint to a JSON file.

        Args:
            path: Destination file path. The file is created or overwritten.
        """
        payload: dict[str, Any] = {
            "version": self.version,
            "agent_name": self.agent_name,
            "stream_id": str(self.stream_id),
            "created_at": self.created_at.isoformat(),
            "variables": self.variables,
            "events": self.events,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Checkpoint":
        """Load a checkpoint from a JSON file.

        Args:
            path: Path to a file previously written by ``Checkpoint.save()``.

        Returns:
            A ``Checkpoint`` ready to be passed to ``MemoryStream.from_checkpoint()``.

        Raises:
            ValueError: If the file's version is not supported.
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        version = payload.get("version", _VERSION)
        if version != _VERSION:
            raise ValueError(f"Checkpoint version {version!r} is not supported (expected {_VERSION!r}).")
        return cls(
            agent_name=payload["agent_name"],
            stream_id=UUID(payload["stream_id"]),
            events=payload["events"],
            variables=payload.get("variables", {}),
            created_at=datetime.fromisoformat(payload["created_at"]),
            version=version,
        )

    def __repr__(self) -> str:
        return (
            f"Checkpoint(agent_name={self.agent_name!r}, "
            f"events={len(self.events)}, "
            f"created_at={self.created_at.isoformat()!r})"
        )
