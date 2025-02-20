# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from .deep_research import DeepResearchAgent
from .discord import DiscordAgent
from .document_agent import DoclingDocIngestAgent, DocumentAgent
from .slack import SlackAgent
from .telegram import TelegramAgent
from .websurfer import WebSurferAgent

__all__ = [
    "DeepResearchAgent",
    "DiscordAgent",
    "DoclingDocIngestAgent",
    "DocumentAgent",
    "SlackAgent",
    "TelegramAgent",
    "WebSurferAgent",
]
