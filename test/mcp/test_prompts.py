# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from mcp.server.lowlevel import NotificationOptions

from ag2 import Agent
from ag2.mcp import MCPServer, Prompt, PromptArgument, PromptMessage
from ag2.mcp.errors import MCPPromptNotFoundError
from ag2.mcp.prompts import PromptProvider
from ag2.testing import TestConfig


@pytest.mark.asyncio
class TestPromptGet:
    async def test_string_render_becomes_user_message(self) -> None:
        provider = PromptProvider([Prompt(name="greet", render=lambda a: f"Hello {a['name']}")])

        result = await provider.get("greet", {"name": "Sam"})

        [message] = result.messages
        assert message.role == "user"
        assert message.content.text == "Hello Sam"

    async def test_async_render(self) -> None:
        async def _render(_a: dict[str, str]) -> str:
            return "async-prompt"

        provider = PromptProvider([Prompt(name="p", render=_render)])

        result = await provider.get("p", {})

        assert result.messages[0].content.text == "async-prompt"

    async def test_explicit_message_sequence(self) -> None:
        provider = PromptProvider([
            Prompt(
                name="chat",
                render=lambda _a: [
                    PromptMessage(role="assistant", text="Hi!"),
                    PromptMessage(role="user", text="Continue"),
                ],
            )
        ])

        result = await provider.get("chat", {})

        assert [(m.role, m.content.text) for m in result.messages] == [("assistant", "Hi!"), ("user", "Continue")]

    async def test_unknown_prompt_raises(self) -> None:
        provider = PromptProvider([])

        with pytest.raises(MCPPromptNotFoundError):
            await provider.get("missing", {})


class TestPromptCapability:
    def test_advertised_only_when_prompts_present(self) -> None:
        agent = Agent("a", config=TestConfig("hi"))
        opts = NotificationOptions()

        without = MCPServer(agent).server.get_capabilities(opts, {})
        with_prompts = MCPServer(agent, prompts=[Prompt(name="greet", render=lambda _a: "hi")]).server.get_capabilities(
            opts, {}
        )

        assert without.prompts is None
        assert with_prompts.prompts is not None


def test_prompt_argument_declaration() -> None:
    prompt = Prompt(
        name="greet",
        render=lambda a: f"Hi {a['name']}",
        arguments=(PromptArgument(name="name", description="who", required=True),),
    )

    assert prompt.arguments[0].name == "name"
    assert prompt.arguments[0].required is True
