# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from xai_sdk.chat import Response as XAIResponse

from autogen.beta.events import BaseEvent, Field


class XAIAssistantEvent(BaseEvent):
    """Carries the raw xAI ``GetChatCompletionResponse`` proto for multi-turn round-trip.

    The xai-sdk ``Chat`` object is stateful: rebuilding an assistant turn with
    ``tool_calls`` requires passing the original ``Response`` proto back via
    ``chat.append(response)``. The proto is the only canonical source — there is
    no public helper that constructs an assistant message with tool_calls from
    primitives. Persisted (NOT transient) so it survives history storage.
    """

    proto_bytes: bytes = Field(repr=False)
    model: str | None = Field(default=None, compare=False)
    finish_reason: str | None = Field(default=None, compare=False)

    @classmethod
    def from_response(cls, response: XAIResponse) -> "XAIAssistantEvent":
        return cls(
            proto_bytes=response.proto.SerializeToString(),
            model=getattr(response, "model", None),
            finish_reason=getattr(response, "finish_reason", None),
        )
