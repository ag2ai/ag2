# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

from ag2.annotations import Context
from ag2.events import Usage
from ag2.exceptions import HumanInputError
from ag2.scopes import derive_child_variables
from ag2.stream import MemoryStream, Stream

from ._forwarding import Forwarder, get_registry

if TYPE_CHECKING:
    from ag2.agent import Agent


@dataclass
class TaskResult:
    task_id: str
    objective: str
    result: str | None
    completed: bool
    stream: Stream
    usage: Usage
    """Cumulative stream snapshot for inspection, never an additive record."""
    error: Exception | None = None
    cancelled: bool = False


async def run_task(
    agent: "Agent",
    objective: str,
    *,
    parent_context: Context,
    context: str = "",
    stream: Stream | None = None,
    emit_events: bool = True,
    task_id: str | None = None,
) -> TaskResult:
    """Run a delegation on its own stream, forwarding all source events upward.

    ``emit_events`` suppresses only this delegation's lifecycle, never its usage.
    Source usage is delivered immediately; no subtask rollup is generated.
    """
    task_id = task_id or uuid4().hex
    task_stream = stream or MemoryStream(storage=parent_context.stream.history.storage)
    prompt = f"{objective}\n\n## Context\n{context}" if context else objective
    forwarder = Forwarder(parent_context, task_id, agent.name, task_stream, objective, emit_events)
    registry = get_registry(parent_context)
    registry.register(forwarder)
    dependencies = parent_context.dependencies.copy()
    dependencies[Forwarder] = forwarder
    try:
        reply = await agent.ask(
            prompt,
            stream=task_stream,
            dependencies=dependencies,
            variables=derive_child_variables(parent_context.variables),
        )
    except asyncio.CancelledError as exc:
        await forwarder.reject(exc)
        origin = forwarder.origin
        reason = origin.reason if origin is not None and origin.halted else "parent cancelled"
        usage = await forwarder.usage()
        task = asyncio.current_task()
        cancelling = getattr(task, "cancelling", None)
        if origin is None or not origin.halted or cancelling is not None and cancelling():
            raise
        return TaskResult(task_id, objective, None, False, task_stream, usage, RuntimeError(reason), cancelled=True)
    except Exception as exc:
        await forwarder.reject(exc)
        usage = await forwarder.usage()
        if isinstance(exc, HumanInputError):
            raise
        return TaskResult(task_id, objective, None, False, task_stream, usage, exc)
    else:
        usage = await forwarder.usage()
        return TaskResult(task_id, objective, reply.body, True, task_stream, usage)
    finally:
        forwarder.close()
        registry.unregister(task_id)
