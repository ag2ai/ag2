# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from  https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
# !/usr/bin/env python3 -m pytest

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from autogen.agentchat.contrib.llamaindex_conversable_agent import LLamaIndexConversableAgent
from autogen.import_utils import optional_import_block, run_for_optional_imports

with optional_import_block() as result:
    from llama_index.core.base.llms.types import ChatMessage  # noqa: F401 — used for typing in fixtures
    from llama_index.core.chat_engine.types import AgentChatResponse


@run_for_optional_imports(["llama_index"], "neo4j")
class TestLLamaIndexConversableAgentLegacy:
    """Cover the legacy llama-index <= 0.12 path where the wrapped agent
    exposes synchronous `chat` and asynchronous `achat` methods that return
    an `AgentChatResponse`."""

    def _agent(self, mock_runner: MagicMock) -> LLamaIndexConversableAgent:
        return LLamaIndexConversableAgent(
            "trip_specialist",
            llama_index_agent=mock_runner,
            system_message="You help customers find places to visit.",
            description="Helps with trip planning.",
            llm_config=False,
        )

    def test_sync_reply_uses_chat_when_available(self) -> None:
        runner = MagicMock()
        runner.chat.return_value = AgentChatResponse(response="The Ghibli Museum in Tokyo is a great fit.")

        agent = self._agent(runner)
        success, reply = agent._generate_oai_reply(
            messages=[{"role": "user", "content": "Where can I see Ghibli art?"}]
        )

        assert success is True
        assert reply == "The Ghibli Museum in Tokyo is a great fit."
        runner.chat.assert_called_once()

    def test_async_reply_uses_achat_when_available(self) -> None:
        runner = MagicMock()
        runner.achat = AsyncMock(return_value=AgentChatResponse(response="See the Studio Ghibli archives."))

        agent = self._agent(runner)
        success, reply = asyncio.run(
            agent._a_generate_oai_reply(messages=[{"role": "user", "content": "Ghibli archives?"}])
        )

        assert success is True
        assert reply == "See the Studio Ghibli archives."
        runner.achat.assert_awaited_once()


@run_for_optional_imports(["llama_index"], "neo4j")
class TestLLamaIndexConversableAgentWorkflow:
    """Cover the llama-index 0.13+ path where the wrapped agent is a
    workflow-based agent (FunctionAgent / ReActAgent / CodeActAgent) and
    only exposes `run(user_msg=..., chat_history=...)`. The legacy `chat`
    and `achat` methods were removed on this branch, so feature-detection
    has to fall through to the workflow path. Regression for #2053."""

    def _make_workflow_agent(self, response_text: str) -> MagicMock:
        """Build a mock that quacks like a 0.13+ workflow agent: no `chat`
        / `achat` surface, but a `run()` that returns a coroutine resolving
        to an `AgentOutput`-shaped object with `.response.content`."""
        mock_agent = MagicMock(spec_set=["run"])  # critically, no `chat` / `achat`
        agent_output = SimpleNamespace(response=SimpleNamespace(content=response_text))

        async def _fake_run(**kwargs: object) -> SimpleNamespace:
            return agent_output

        mock_agent.run.side_effect = _fake_run
        return mock_agent

    def _agent(self, mock_workflow: MagicMock) -> LLamaIndexConversableAgent:
        return LLamaIndexConversableAgent(
            "trip_specialist",
            llama_index_agent=mock_workflow,
            system_message="You help customers find places to visit.",
            description="Helps with trip planning.",
            llm_config=False,
        )

    def test_sync_reply_drives_workflow_run(self) -> None:
        mock_workflow = self._make_workflow_agent("Visit teamLab Planets in Toyosu.")
        agent = self._agent(mock_workflow)

        success, reply = agent._generate_oai_reply(
            messages=[{"role": "user", "content": "Where can I see digital art in Tokyo?"}]
        )

        assert success is True
        assert reply == "Visit teamLab Planets in Toyosu."
        mock_workflow.run.assert_called_once()
        _, kwargs = mock_workflow.run.call_args
        assert kwargs["user_msg"] == "Where can I see digital art in Tokyo?"
        # chat_history is forwarded but empty for a single-turn invocation.
        assert kwargs["chat_history"] == []

    def test_async_reply_drives_workflow_run(self) -> None:
        mock_workflow = self._make_workflow_agent("Try the Imperial Palace East Gardens.")
        agent = self._agent(mock_workflow)

        success, reply = asyncio.run(
            agent._a_generate_oai_reply(messages=[{"role": "user", "content": "Quiet spot in Tokyo?"}])
        )

        assert success is True
        assert reply == "Try the Imperial Palace East Gardens."
        mock_workflow.run.assert_called_once()

    def test_workflow_response_plain_string_is_passed_through(self) -> None:
        """Older preview builds returned a bare string on `.response` rather
        than a `ChatMessage`. The wrapper must still produce a string and
        not the object's repr."""
        mock_agent = MagicMock(spec_set=["run"])
        agent_output = SimpleNamespace(response="bare string reply")

        async def _fake_run(**kwargs: object) -> SimpleNamespace:
            return agent_output

        mock_agent.run.side_effect = _fake_run

        agent = self._agent(mock_agent)
        success, reply = agent._generate_oai_reply(messages=[{"role": "user", "content": "Anything?"}])

        assert success is True
        assert reply == "bare string reply"


def test_llama_index_agent_required() -> None:
    """No optional import gate: argument validation happens before the
    llama-index import is required."""
    with pytest.raises(ValueError, match="llama_index_agent must be provided"):
        LLamaIndexConversableAgent("agent", llama_index_agent=None, description="x")
