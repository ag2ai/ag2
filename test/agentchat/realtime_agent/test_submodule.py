# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT


def test_import() -> None:
    from autogen.agentchat.realtime.experimental import RealtimeAgent, RealtimeObserver

    assert RealtimeAgent is not None
    assert RealtimeObserver is not None


def test_import_clients() -> None:
    from autogen.agentchat.realtime.experimental.clients import (
        GeminiRealtimeClient,
        OpenAIRealtimeClient,
        RealtimeClientProtocol,
        Role,
    )

    llm_config = {"config_list": [{"model": "gpt-4o", "api_key": "some_key", "base_url": "some url"}]}
    gemini_client = GeminiRealtimeClient(llm_config=llm_config)
    openai_client = OpenAIRealtimeClient(llm_config=llm_config)

    assert RealtimeClientProtocol is not None
    assert Role is not None
    assert isinstance(gemini_client, RealtimeClientProtocol)
    assert isinstance(openai_client, RealtimeClientProtocol)
