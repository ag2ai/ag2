# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A deterministic world and a context-sensitive fake model.

The fake model only knows what its context shows it, which is what makes a
compaction observable without a real LLM:

* it needs the session token from ``login`` to ``fetch``; if the token is not in
  its context it logs in again (a refetch), or — with ``guess_token`` — calls
  ``fetch`` with a made-up token (a blocked call);
* it fetches the next item after the ones its context shows fetched, so a
  context that lost earlier fetches makes it fetch them again (refetches).
"""

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from typing_extensions import Self

from ag2 import Agent, Context, MemoryStream, tool
from ag2.compact import CompactionSummary
from ag2.config import LLMClient, ModelConfig, ModelProvider
from ag2.events import (
    BaseEvent,
    ModelMessage,
    ModelResponse,
    ToolCallEvent,
    ToolCallsEvent,
    ToolResultsEvent,
    render_for_prompt,
)
from ag2.extensions.compaction_verifier import ReplayEnvironment
from ag2.testing import TestConfig

ITEMS = ("a", "b", "c", "d", "e", "f", "g", "h")
TOKEN = "tok-7f3a"


class World:
    """A login-then-fetch service with deterministic state."""

    def __init__(self) -> None:
        self.logins = 0
        self.fetched: list[str] = []

    def login(self, user: str) -> str:
        """Log in and return a session token."""
        self.logins += 1
        return TOKEN

    def fetch(self, token: str, item: str) -> str:
        """Fetch one item with a session token."""
        if token != TOKEN:
            raise PermissionError("invalid session token")
        if item not in ITEMS:
            raise KeyError(item)
        self.fetched.append(item)
        return f"value of {item}"


def world_tools() -> dict[str, Callable[..., Any]]:
    world = World()
    return {"login": world.login, "fetch": world.fetch}


def world_environment() -> ReplayEnvironment:
    return ReplayEnvironment(world_tools)


class _ContextAwareClient(LLMClient):
    def __init__(self, guess_token: bool) -> None:
        self._guess_token = guess_token
        self._calls = 0

    async def __call__(self, messages: Sequence[BaseEvent], context: Context, **kwargs: Any) -> ModelResponse:
        self._calls += 1
        # Only what tools returned (or a summary of it) counts as known: the
        # model knows the token if it saw login's output.
        text = "\n".join(render_for_prompt(m) for m in messages if isinstance(m, (ToolResultsEvent, CompactionSummary)))
        fetched = set(re.findall(r"value of (\w)", text))
        remaining = [i for i in ITEMS if i not in fetched]
        if not remaining:
            return ModelResponse(ModelMessage("all items fetched"))
        call_id = f"call-{id(self)}-{self._calls}"
        if TOKEN in text:
            args = {"token": TOKEN, "item": remaining[0]}
            return ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent("fetch", arguments=json.dumps(args), id=call_id)])
            )
        if self._guess_token:
            args = {"token": "tok-guess", "item": remaining[0]}
            return ModelResponse(
                tool_calls=ToolCallsEvent([ToolCallEvent("fetch", arguments=json.dumps(args), id=call_id)])
            )
        return ModelResponse(
            tool_calls=ToolCallsEvent([ToolCallEvent("login", arguments=json.dumps({"user": "ada"}), id=call_id)])
        )


class ContextAwareConfig(ModelConfig):
    """A fake model whose next call depends only on what its context shows."""

    def __init__(self, *, guess_token: bool = False) -> None:
        self._guess_token = guess_token

    @property
    def provider(self) -> ModelProvider:
        raise NotImplementedError

    @property
    def model(self) -> str:
        return "context-aware-fake"

    def copy(self) -> Self:
        return self

    def create(self) -> LLMClient:
        return _ContextAwareClient(self._guess_token)

    def create_files_client(self) -> Any:
        raise NotImplementedError


async def record_trajectory(fetches: int = 6) -> list[BaseEvent]:
    """A real AG2 history: log in once, then fetch ``fetches`` items, then answer."""
    turns: list[Any] = [ToolCallEvent("login", arguments='{"user": "ada"}', id="rec-0")]
    turns += [
        ToolCallEvent("fetch", arguments=json.dumps({"token": TOKEN, "item": ITEMS[i]}), id=f"rec-{i + 1}")
        for i in range(fetches)
    ]
    turns.append("done")
    agent = Agent("recorder", config=TestConfig(*turns, raise_tool_errors=False))
    stream = MemoryStream()
    await agent.ask("Fetch the items.", stream=stream, tools=[tool(f, name=n) for n, f in world_tools().items()])
    return list(await stream.history.get_events())
