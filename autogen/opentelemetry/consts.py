# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Based on OpenTelemetry GenAI semantic conventions
# https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/

from enum import Enum

from autogen.version import __version__ as AG2_VERSION  # noqa: N812


class SpanType(str, Enum):
    CONVERSATION = "conversation"  # Initiate Chat / Run Chat
    MULTI_CONVERSATION = "multi_conversation"  # Initiate Chats (sequential/parallel)
    AGENT = "agent"  # Agent's Generate Reply (invoke_agent)
    LLM = "llm"  # LLM Invocation (chat completion)
    TOOL = "tool"  # Tool Execution (execute_tool)
    HANDOFF = "handoff"  # Handoff (TODO)
    SPEAKER_SELECTION = "speaker_selection"  # Group Chat Speaker Selection
    HUMAN_INPUT = "human_input"  # Human-in-the-loop input (await_human_input)
    CODE_EXECUTION = "code_execution"  # Code execution (execute_code_blocks)


OTEL_SCHEMA = "https://opentelemetry.io/schemas/1.11.0"
INSTRUMENTING_MODULE_NAME = "opentelemetry.instrumentation.ag2"
INSTRUMENTING_LIBRARY_VERSION = AG2_VERSION
