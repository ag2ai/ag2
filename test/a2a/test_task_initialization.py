# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Test that task variable is properly initialized in RemoteAgent.a_generate_remote_reply.

Regression test for issue #2223: UnboundLocalError when task variable is accessed
before assignment in error scenarios.
"""

from unittest.mock import AsyncMock

import pytest

from autogen import ConversableAgent
from autogen.a2a import A2aRemoteAgent, MockClient


@pytest.mark.asyncio
async def test_message_only_flow_no_unbound_task():
    """Test that message-only flow (no Task events) doesn't cause UnboundLocalError.

    MockClient returns a SendMessageSuccessResponse containing a Message, so the
    stream yields only Message events. The 'task' variable in the else branch is
    never assigned. The fix ensures task is initialized to None before the event
    loop, preventing UnboundLocalError in error handlers or debugging code.
    """
    remote_agent = A2aRemoteAgent(
        url="http://fake",
        name="message-only-agent",
        client=MockClient("Response without task events"),
    )

    client_agent = AsyncMock(spec=ConversableAgent)
    client_agent.silent = True
    client_agent.name = "test_client"

    # This should complete without UnboundLocalError
    await remote_agent.a_receive("test message", client_agent, request_reply=True)

    # Verify we got a valid response in message history
    history = remote_agent.chat_messages[client_agent]
    assert len(history) == 2
    assert history[0]["content"] == "test message"
    assert history[1]["content"] == "Response without task events"
