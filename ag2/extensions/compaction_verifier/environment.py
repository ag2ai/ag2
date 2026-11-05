# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Environments: rebuilding the world an agent's tools act on.

Both arms of a boundary must start from the state the environment was in at
the cut, or a difference between them measures the environment, not the
context. AG2 cannot snapshot what arbitrary tools did to the world, so the
verifier asks for it through :class:`Environment`.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from ag2.annotations import Context
from ag2.context import ConversationContext
from ag2.events import ToolCallEvent, ToolErrorEvent
from ag2.stream import MemoryStream
from ag2.tools import tool
from ag2.tools.final import FunctionTool
from ag2.tools.tool import Tool

from .actions import Action

__all__ = ("Environment", "ReplayEnvironment", "ReplayMismatchError")


@runtime_checkable
class Environment(Protocol):
    """Rebuilds the environment in the state a sequence of executed calls left it in."""

    async def restore(self, prefix: Sequence[Action]) -> Sequence[Tool]:
        """Return tools bound to a fresh environment into which ``prefix`` has been applied.

        Called once per rollout, so every rollout gets its own environment and
        rollouts can run concurrently. The tool names must be the names the
        recorded calls used; wrap plain functions with :func:`ag2.tool`.
        """
        ...


class ReplayMismatchError(RuntimeError):
    """A replayed call succeeded where the recording failed, or the reverse."""


class ReplayEnvironment:
    """An :class:`Environment` whose state is rebuilt by re-executing the recorded calls.

    ``factory`` returns a fresh ``{tool name: function}`` mapping bound to new
    state every time it is called. Each function is wrapped with
    :func:`ag2.tool`, and each recorded call is replayed through that tool with
    its recorded arguments — the same validation and coercion the agent's own
    executor applied when the call was recorded. Results are discarded; a call
    that failed when it was recorded is expected to fail again.

    This is correct when the functions are deterministic given their arguments
    and their state — no clocks, randomness or external services. Replay checks
    that assumption as far as it can: if a call's outcome (success or failure)
    differs from the recording, :class:`ReplayMismatchError` is raised rather
    than letting a rollout start from the wrong state. For anything else,
    implement :class:`Environment` directly, for example with a snapshot.

    Example::

        def fresh_world() -> dict[str, Callable[..., Any]]:
            world = World(seed=7)
            return {"search": world.search, "update": world.update}


        environment = ReplayEnvironment(fresh_world)
    """

    def __init__(self, factory: Callable[[], Mapping[str, Callable[..., Any]]]) -> None:
        self._factory = factory

    async def restore(self, prefix: Sequence[Action]) -> Sequence[Tool]:
        tools = {name: tool(function, name=name) for name, function in self._factory().items()}
        context: Context = ConversationContext(MemoryStream())
        for action in prefix:
            await _replay(tools, action, context)
        return list(tools.values())


async def _replay(tools: Mapping[str, FunctionTool], action: Action, context: Context) -> None:
    replayed = tools.get(action.name)
    if replayed is None:
        if action.blocked:
            return  # the recorded call named a tool that does not exist; it changed nothing
        raise ReplayMismatchError(f"no function for recorded call {action.signature}")
    result = await replayed(ToolCallEvent(action.name, arguments=action.arguments, id=action.call_id), context)
    if isinstance(result, ToolErrorEvent) and not action.blocked:
        raise ReplayMismatchError(
            f"recorded call {action.signature} succeeded but raised {type(result.error).__name__} on replay"
        ) from result.error
    if not isinstance(result, ToolErrorEvent) and action.blocked:
        raise ReplayMismatchError(f"recorded call {action.signature} failed but succeeded on replay")
