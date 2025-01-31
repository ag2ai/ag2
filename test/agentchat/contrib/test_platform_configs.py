import pytest
from typing import Dict, Optional
from pydantic import BaseModel, Field, ValidationError

from autogen.agentchat.contrib.comms.platform_configs import (
    ReplyMonitorConfig,
    BaseCommsPlatformConfig
)

def test_reply_monitor_config_init():
    config = ReplyMonitorConfig(timeout_minutes=5, max_reply_messages=10)
    assert config.timeout_minutes == 5
    assert config.max_reply_messages == 10

def test_reply_monitor_config_defaults():
    config = ReplyMonitorConfig()
    assert config.timeout_minutes == 1
    assert config.max_reply_messages == 1

def test_reply_monitor_config_validation():
    config = ReplyMonitorConfig(timeout_minutes=5, max_reply_messages=10)
    assert config.timeout_minutes == 5
    assert config.max_reply_messages == 10

from pydantic import ValidationError

def test_reply_monitor_config_invalid_timeout():
    with pytest.raises(ValidationError):
        ReplyMonitorConfig(timeout_minutes=-1)

def test_reply_monitor_config_invalid_max_replies():
    with pytest.raises(ValidationError):
        ReplyMonitorConfig(max_reply_messages=0)

def test_platform_specific_configs():
    from autogen.agentchat.contrib.comms.telegram_agent import TelegramConfig
    from autogen.agentchat.contrib.comms.slack_agent import SlackConfig 
    from autogen.agentchat.contrib.comms.discord_agent import DiscordConfig

    # Test Telegram config
    telegram_config = TelegramConfig(
        bot_token="123456:ABC-DEF",
        destination_id="@testchannel"
    )
    assert telegram_config.bot_token == "123456:ABC-DEF"
    assert telegram_config.destination_id == "@testchannel"

    # Test Slack config
    slack_config = SlackConfig(
        bot_token="xoxb-fake",
        app_token="xapp-fake",
        channel_id="ABC123",
        signing_secret="secret"
    )
    assert slack_config.bot_token == "xoxb-fake"
    assert slack_config.app_token == "xapp-fake"
    assert slack_config.channel_id == "ABC123"
    assert slack_config.signing_secret == "secret"

    # Test Discord config
    discord_config = DiscordConfig(
        bot_token="fake_token",
        guild_name="TestGuild",
        channel_name="TestChannel"
    )
    assert discord_config.bot_token == "fake_token"
    assert discord_config.guild_name == "TestGuild"
    assert discord_config.channel_name == "TestChannel"
