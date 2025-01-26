# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai OSS project maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT


def test_import() -> None:
    from autogen.agentchat.realtime_agent import RealtimeAgent, RealtimeObserver

    assert RealtimeAgent is not None
    assert RealtimeObserver is not None


def test_import_clients() -> None:
    from autogen.agentchat.realtime_agent.clients import (
        GeminiRealtimeClient,
        OpenAIRealtimeClient,
        RealtimeClientProtocol,
        Role,
    )

    assert RealtimeClientProtocol is not None
    assert Role is not None
    assert issubclass(GeminiRealtimeClient, RealtimeClientProtocol)
    assert issubclass(OpenAIRealtimeClient, RealtimeClientProtocol)
