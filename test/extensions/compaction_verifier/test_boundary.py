# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.annotations import Context
from ag2.compact import TailWindowCompact
from ag2.events import BaseEvent, ToolResultsEvent, Usage, UsageEvent
from ag2.extensions.compaction_verifier import (
    IdentityCompact,
    compact_context,
    make_boundary,
    resumable_cuts,
    select_cuts,
)
from ag2.knowledge import KnowledgeStore

from .conftest import record_trajectory


class _DropEverything:
    async def compact(self, events: list[BaseEvent], context: Context, store: KnowledgeStore | None) -> list[BaseEvent]:
        return []


class _SpendsTokens:
    async def compact(self, events: list[BaseEvent], context: Context, store: KnowledgeStore | None) -> list[BaseEvent]:
        await context.send(UsageEvent(Usage(prompt_tokens=100, completion_tokens=20), kind="compaction"))
        return list(events)


class TestCuts:
    @pytest.mark.asyncio
    async def test_a_cut_follows_every_tool_results_event(self) -> None:
        events = await record_trajectory(fetches=6)

        cuts = resumable_cuts(events)

        assert len(cuts) == 7
        assert all(isinstance(events[c - 1], ToolResultsEvent) for c in cuts)

    @pytest.mark.asyncio
    async def test_select_cuts_spaces_by_executed_actions_and_skips_the_end(self) -> None:
        events = await record_trajectory(fetches=6)  # 7 actions

        cuts = select_cuts(events, every=2, min_prefix=3)

        assert [len(make_boundary(events, c).prefix) for c in cuts] == [3, 5]

    def test_select_cuts_rejects_non_positive_spacing(self) -> None:
        with pytest.raises(ValueError):
            select_cuts([], every=0)


class TestMakeBoundary:
    @pytest.mark.asyncio
    async def test_prefix_and_history(self) -> None:
        events = await record_trajectory(fetches=4)
        cut = resumable_cuts(events)[2]

        boundary = make_boundary(events, cut, trajectory=3)

        assert boundary.trajectory == 3
        assert [a.name for a in boundary.prefix] == ["login", "fetch", "fetch"]
        assert 'login({"user":"ada"})' in boundary.history
        assert boundary.history_tools == {"login", "fetch"}
        assert boundary.context[-1] is events[cut - 1]

    @pytest.mark.asyncio
    async def test_a_cut_that_is_not_after_tool_results_is_rejected(self) -> None:
        events = await record_trajectory(fetches=2)

        with pytest.raises(ValueError, match="ToolResultsEvent"):
            make_boundary(events, 1)


class TestCompactContext:
    @pytest.mark.asyncio
    async def test_tail_window_shrinks_the_context_and_keeps_the_trigger(self) -> None:
        events = await record_trajectory(fetches=6)
        boundary = make_boundary(events, resumable_cuts(events)[5])

        # one tool round is five conversational events; a window of five keeps the last round
        post = await compact_context(TailWindowCompact(target=5), boundary)

        assert post.events[-1] == boundary.context[-1]
        assert post.size.events < boundary.size.events
        assert post.size.tokens < boundary.size.tokens

    @pytest.mark.asyncio
    async def test_a_window_narrower_than_a_tool_round_keeps_everything(self) -> None:
        # AG2 behaviour the verifier surfaces rather than hides: the window's start
        # lands mid-round, snap() advances it past the end, and the strategy keeps
        # the whole history.
        events = await record_trajectory(fetches=6)
        boundary = make_boundary(events, resumable_cuts(events)[5])

        post = await compact_context(TailWindowCompact(target=2), boundary)

        assert post.events == boundary.context

    @pytest.mark.asyncio
    async def test_identity_keeps_everything(self) -> None:
        events = await record_trajectory(fetches=3)
        boundary = make_boundary(events, resumable_cuts(events)[-1])

        post = await compact_context(IdentityCompact(), boundary)

        assert post.events == boundary.context
        assert post.usage == ()

    @pytest.mark.asyncio
    async def test_a_strategy_that_drops_the_trigger_is_rejected(self) -> None:
        events = await record_trajectory(fetches=2)
        boundary = make_boundary(events, resumable_cuts(events)[-1])

        with pytest.raises(ValueError, match="same trigger"):
            await compact_context(_DropEverything(), boundary)

    @pytest.mark.asyncio
    async def test_the_strategy_spend_is_collected(self) -> None:
        events = await record_trajectory(fetches=2)
        boundary = make_boundary(events, resumable_cuts(events)[-1])

        post = await compact_context(_SpendsTokens(), boundary)

        assert [u.usage.prompt_tokens for u in post.usage] == [100]
