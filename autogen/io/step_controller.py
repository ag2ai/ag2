# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Step-by-step execution controller for run() and run_group_chat()."""

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..events.base_event import BaseEvent


class StepController:
    """Controls step-by-step execution for synchronous runs.

    When step mode is enabled, the producer (background thread) blocks after
    sending each event until the consumer calls step() to acknowledge it.

    Args:
        yield_on: List of event types to yield on. If None, yields on every event.
    """

    def __init__(self, yield_on: Sequence[type["BaseEvent"]] | None = None):
        self._yield_on: tuple[type[BaseEvent], ...] | None = tuple(yield_on) if yield_on else None
        self._step_event = threading.Event()
        self._terminated = False

    def should_block(self, event: "BaseEvent") -> bool:
        """Check if we should block on this event."""
        if self._terminated:
            return False
        # If no filter, yield on every event
        if self._yield_on is None:
            return True
        # Otherwise, yield only on specified event types
        return isinstance(event, self._yield_on)

    def wait_for_step(self, event: "BaseEvent") -> None:
        """Called by producer after putting event. Blocks if should_block returns True."""
        if not self.should_block(event):
            return
        self._step_event.clear()
        self._step_event.wait()

    def step(self) -> None:
        """Called by consumer to advance to next event."""
        self._step_event.set()

    def terminate(self) -> None:
        """Unblock producer for shutdown."""
        self._terminated = True
        self._step_event.set()


class AsyncStepController:
    """Controls step-by-step execution for async runs.

    When step mode is enabled, the producer (background task) blocks after
    sending each event until the consumer calls step() to acknowledge it.

    Args:
        yield_on: List of event types to yield on. If None, yields on every event.
    """

    def __init__(self, yield_on: Sequence[type["BaseEvent"]] | None = None):
        self._yield_on: tuple[type[BaseEvent], ...] | None = tuple(yield_on) if yield_on else None
        self._step_event = asyncio.Event()
        self._terminated = False

    def should_block(self, event: "BaseEvent") -> bool:
        """Check if we should block on this event."""
        if self._terminated:
            return False
        if self._yield_on is None:
            return True
        return isinstance(event, self._yield_on)

    async def wait_for_step(self, event: "BaseEvent") -> None:
        """Called by producer after putting event. Blocks if should_block returns True."""
        if not self.should_block(event):
            return
        self._step_event.clear()
        await self._step_event.wait()

    def step(self) -> None:
        """Called by consumer to advance to next event."""
        self._step_event.set()

    def terminate(self) -> None:
        """Unblock producer for shutdown."""
        self._terminated = True
        self._step_event.set()
