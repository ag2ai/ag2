# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.events import ModelMessage
from autogen.beta.network.primitives.envelope import Envelope
from autogen.beta.network.primitives.priority import (
    DefaultPriority,
    DefaultPriorityScheme,
    HighestPriorityWins,
)


class TestDefaultPriority:
    def test_ordering(self) -> None:
        assert DefaultPriority.BACKGROUND < DefaultPriority.NORMAL
        assert DefaultPriority.NORMAL < DefaultPriority.URGENT

    def test_int_values(self) -> None:
        assert int(DefaultPriority.BACKGROUND) == 0
        assert int(DefaultPriority.NORMAL) == 1
        assert int(DefaultPriority.URGENT) == 2


class TestDefaultPriorityScheme:
    def test_compare_higher(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(DefaultPriority.URGENT, DefaultPriority.BACKGROUND) > 0

    def test_compare_lower(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(DefaultPriority.BACKGROUND, DefaultPriority.URGENT) < 0

    def test_compare_equal(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(DefaultPriority.NORMAL, DefaultPriority.NORMAL) == 0

    def test_compare_with_ints(self) -> None:
        scheme = DefaultPriorityScheme()
        assert scheme.compare(2, 0) > 0
        assert scheme.compare(0, 2) < 0


class TestHighestPriorityWins:
    @pytest.mark.asyncio
    async def test_higher_incoming_wins(self) -> None:
        resolver = HighestPriorityWins()
        existing = Envelope(
            event=ModelMessage(content="low"),
            sender="a",
            priority=DefaultPriority.BACKGROUND,
        )
        incoming = Envelope(
            event=ModelMessage(content="high"),
            sender="b",
            priority=DefaultPriority.URGENT,
        )
        winner = await resolver.resolve(existing, incoming)
        assert winner is incoming

    @pytest.mark.asyncio
    async def test_lower_incoming_loses(self) -> None:
        resolver = HighestPriorityWins()
        existing = Envelope(
            event=ModelMessage(content="high"),
            sender="a",
            priority=DefaultPriority.URGENT,
        )
        incoming = Envelope(
            event=ModelMessage(content="low"),
            sender="b",
            priority=DefaultPriority.BACKGROUND,
        )
        winner = await resolver.resolve(existing, incoming)
        assert winner is existing

    @pytest.mark.asyncio
    async def test_equal_priority_keeps_existing(self) -> None:
        resolver = HighestPriorityWins()
        existing = Envelope(
            event=ModelMessage(content="first"),
            sender="a",
            priority=DefaultPriority.NORMAL,
        )
        incoming = Envelope(
            event=ModelMessage(content="second"),
            sender="b",
            priority=DefaultPriority.NORMAL,
        )
        winner = await resolver.resolve(existing, incoming)
        assert winner is existing
