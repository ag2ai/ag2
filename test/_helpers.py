# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.events import BaseEvent, ModelReasoning, ProviderReplay
from ag2.tools import tool


class DurableReasoning(ModelReasoning, ProviderReplay):
    """Provider reasoning item that must be replayed, like OpenAIReasoningEvent.

    Reasoning is transient by default; a provider that persists its item opts out
    (see ``ag2.config.openai.events``). Redeclared here so tests outside a
    provider package can exercise the anchor case without importing its SDK.
    """

    __transient__ = False
    __replay_role__ = "anchor"


class ProviderTurnState(BaseEvent, ProviderReplay):
    """Provider-native object standing in for a whole assistant turn.

    Like ``XAIAssistantEvent``: the only way to rebuild a turn carrying
    ``tool_calls`` for a provider whose SDK cannot construct one from primitives.
    Redeclared here so tests outside a provider package can exercise the turn
    case without importing its SDK.
    """

    __replay_role__ = "turn"


@tool
def lookup() -> str:
    """Look something up."""
    return "42"
