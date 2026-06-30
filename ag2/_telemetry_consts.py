# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Span-attribute constants the eval trace-reader matches against.

These mirror the attributes that ``TelemetryMiddleware``
(``ag2.middleware.builtin``) writes onto its spans. The eval trace
reconstructor (``ag2.eval.sources``) reads them back to rebuild a
``Trace`` from OpenTelemetry spans.

This module imports **nothing** — reading these keys never pulls in
OpenTelemetry, so the eval reader stays usable without the tracing extras.
"""

# ── Span-type discriminator ─────────────────────────────────────────────────
# ``ag2.span.type`` + its closed value vocabulary — the field the eval reader
# filters on to decide what kind of event a span represents.
ATTR_SPAN_TYPE = "ag2.span.type"
SPAN_TYPE_AGENT = "agent"  # invoke_agent
SPAN_TYPE_LLM = "llm"  # chat
SPAN_TYPE_TOOL = "tool"  # execute_tool
SPAN_TYPE_HUMAN_INPUT = "human_input"  # await_human_input

# ── Human-input capture ─────────────────────────────────────────────────────
ATTR_HUMAN_INPUT_PROMPT = "ag2.human_input.prompt"
ATTR_HUMAN_INPUT_RESPONSE = "ag2.human_input.response"
