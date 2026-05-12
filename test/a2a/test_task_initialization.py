# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Test that task variable is properly initialized in RemoteAgent.a_generate_remote_reply.

Regression test for issue #2223: UnboundLocalError when task variable is accessed
before assignment in error scenarios.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.compat.v0_3.types import Message, Role

from autogen.a2a import A2aRemoteAgent


class MessageOnlyClient:
    """Mock client that only yields Message events, never Task events."""

    async def send_message(self, message: Message, **kwargs: Any):
        """Yield only Message events, simulating a message-only flow."""
        yield Message(
            role=Role.assistant,
            parts=[{"type": "text", "text": "Response without task events"}],
        )

    async def resubscribe(self, params):
        """Not used in message-only flow."""
        if False:
            yield

    async def get_task(self, params):
        """Should not be called in message-only flow."""
        raise AssertionError("get_task should not be called in message-only flow")


@pytest.mark.asyncio
async def test_message_only_flow_no_unbound_task():
    """Test that message-only flow (no Task events) doesn't cause UnboundLocalError.

    This tests the scenario where all events from the stream are Message types,
    so the 'task' variable in the else branch is never assigned. The fix ensures
    task is initialized to None before the event loop, preventing UnboundLocalError
    in error handlers or debugging code.
    """
    # Create agent with mock client that only yields Message events
    remote_agent = A2aRemoteAgent(
        url="http://test.example.com",
        name="message-only-agent",
        client=MessageOnlyClient(),
    )

    client_agent = AsyncMock()
    client_agent.silent = True
    client_agent.name = "test_client"

    # This should complete without UnboundLocalError
    # The task variable should be initialized to None and never cause issues
    result = await remote_agent.a_generate_remote_reply(
        messages=[{"role": "user", "content": "test"}],
        sender=client_agent,
    )

    # Verify we got a valid response
    assert result is not None
    is_processed, reply = result
    assert is_processed is True
