# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from typing import Any

import pytest

from ag2.extensions.compaction_verifier import (
    Action,
    Environment,
    ReplayEnvironment,
    ReplayMismatchError,
    call_signature,
)
from ag2.tools.tool import Tool

from .conftest import TOKEN, World


def recorded(name: str, blocked: bool = False, **args: Any) -> Action:
    arguments = json.dumps(args)
    return Action(
        call_id=name, name=name, arguments=arguments, signature=call_signature(name, arguments), blocked=blocked
    )


def tracked_environment() -> tuple[ReplayEnvironment, list[World]]:
    worlds: list[World] = []

    def factory() -> dict[str, Callable[..., Any]]:
        world = World()
        worlds.append(world)
        return {"login": world.login, "fetch": world.fetch}

    return ReplayEnvironment(factory), worlds


class TestReplayEnvironment:
    def test_satisfies_the_protocol(self) -> None:
        env, _ = tracked_environment()

        assert isinstance(env, Environment)

    @pytest.mark.asyncio
    async def test_replays_the_prefix_into_fresh_state(self) -> None:
        env, worlds = tracked_environment()
        prefix = [recorded("login", user="ada"), recorded("fetch", token=TOKEN, item="a")]

        tools = await env.restore(prefix)
        await env.restore(prefix)

        assert len(worlds) == 2
        assert worlds[0] is not worlds[1]
        assert worlds[0].fetched == ["a"]
        assert all(isinstance(t, Tool) for t in tools)

    @pytest.mark.asyncio
    async def test_a_recorded_failure_that_fails_again_is_fine(self) -> None:
        env, worlds = tracked_environment()

        await env.restore([recorded("fetch", blocked=True, token="wrong", item="a")])

        assert worlds[0].fetched == []

    @pytest.mark.asyncio
    async def test_a_call_that_now_fails_is_a_mismatch(self) -> None:
        env, _ = tracked_environment()

        with pytest.raises(ReplayMismatchError, match="succeeded but raised PermissionError"):
            await env.restore([recorded("fetch", token="wrong", item="a")])

    @pytest.mark.asyncio
    async def test_a_call_that_now_succeeds_is_a_mismatch(self) -> None:
        env, _ = tracked_environment()

        with pytest.raises(ReplayMismatchError, match="failed but succeeded"):
            await env.restore([recorded("fetch", blocked=True, token=TOKEN, item="a")])

    @pytest.mark.asyncio
    async def test_unknown_tools(self) -> None:
        env, _ = tracked_environment()

        await env.restore([recorded("teleport", blocked=True)])
        with pytest.raises(ReplayMismatchError, match="no function"):
            await env.restore([recorded("teleport")])

    @pytest.mark.asyncio
    async def test_arguments_are_coerced_as_the_agent_executor_does(self) -> None:
        # A model often sends numbers as strings; the tool layer coerces them.
        # Replaying through the tool keeps that, so the state matches the recording.
        pages: list[int] = []

        def page(n: int) -> str:
            pages.append(n + 1)
            return "ok"

        env = ReplayEnvironment(lambda: {"page": page})

        await env.restore([Action(call_id="1", name="page", arguments='{"n": "2"}', signature="page", blocked=False)])

        assert pages == [3]

    @pytest.mark.asyncio
    async def test_async_functions_are_awaited(self) -> None:
        log: list[str] = []

        async def note(text: str) -> str:
            log.append(text)
            return "ok"

        env = ReplayEnvironment(lambda: {"note": note})

        await env.restore([recorded("note", text="hi")])

        assert log == ["hi"]
