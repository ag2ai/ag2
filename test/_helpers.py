# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.events import ModelReasoning


class DurableReasoning(ModelReasoning):
    """Provider reasoning item that must be replayed, like OpenAIReasoningEvent.

    Reasoning is transient by default; a provider that persists its item opts out
    (see ``ag2.config.openai.events``). Redeclared here so tests outside a
    provider package can exercise the durable case without importing its SDK.
    """

    __transient__ = False
