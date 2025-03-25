# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


class TestToolImports:
    def test_imports_experimental(self) -> None:
        """Ensure all tool imports are correct."""
        from autogen.tools.experimental import (
            BrowserUseTool,
            Crawl4AITool,
            DiscordRetrieveTool,
            DiscordSendTool,
            PerplexitySearchTool,
            SlackRetrieveTool,
            SlackSendTool,
            TelegramRetrieveTool,
            TelegramSendTool,
        )

        assert isinstance(BrowserUseTool, type)
        assert isinstance(Crawl4AITool, type)
        assert isinstance(DiscordRetrieveTool, type)
        assert isinstance(DiscordSendTool, type)
        assert isinstance(SlackRetrieveTool, type)
        assert isinstance(SlackSendTool, type)
        assert isinstance(TelegramRetrieveTool, type)
        assert isinstance(TelegramSendTool, type)
        assert isinstance(PerplexitySearchTool, type)

    def test_imports_experimental_messageplatform(self) -> None:
        """Ensure all tool imports are correct."""
        from autogen.tools.experimental.messageplatform import (
            DiscordRetrieveTool,
            DiscordSendTool,
            SlackRetrieveTool,
            SlackSendTool,
            TelegramRetrieveTool,
            TelegramSendTool,
        )

        assert isinstance(DiscordRetrieveTool, type)
        assert isinstance(DiscordSendTool, type)
        assert isinstance(SlackRetrieveTool, type)
        assert isinstance(SlackSendTool, type)
        assert isinstance(TelegramRetrieveTool, type)
        assert isinstance(TelegramSendTool, type)

    def test_imports_experimental_messageplatform_individual(self) -> None:
        pass

    def test_experimental_mesageplatform_all_exports(self) -> None:
        from autogen.tools.experimental.messageplatform import __all__ as messageplatform_all
        from autogen.tools.experimental.messageplatform.discord import __all__ as discord_all
        from autogen.tools.experimental.messageplatform.slack import __all__ as slack_all
        from autogen.tools.experimental.messageplatform.telegram import __all__ as telegram_all

        # Verify each module's __all__ contains expected tools
        assert set(discord_all) == {"DiscordRetrieveTool", "DiscordSendTool"}
        assert set(slack_all) == {"SlackRetrieveTool", "SlackSendTool"}
        assert set(telegram_all) == {"TelegramRetrieveTool", "TelegramSendTool"}
        assert all(
            tool in messageplatform_all
            for tool in (
                "DiscordRetrieveTool",
                "DiscordSendTool",
                "SlackRetrieveTool",
                "SlackSendTool",
                "TelegramRetrieveTool",
                "TelegramSendTool",
            )
        )
