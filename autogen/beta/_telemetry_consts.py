# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenTelemetry string constants — the AG2 beta telemetry vocabulary.

Single source of truth for the telemetry strings that cross package
boundaries:

* the agent-side ``TelemetryMiddleware`` (``autogen.beta.middleware.builtin``),
* the hub-side tracing (``autogen.beta.network.hub._envelope_tracing`` and
  ``...hub.telemetry``),
* the network dispatch handler (``autogen.beta.network.client.handlers``),
* the eval trace reconstructor (``autogen.beta.eval.sources``), which reads
  these keys back to rebuild a ``Trace`` from OpenTelemetry spans.

This module imports **nothing** — importing it never pulls in OpenTelemetry.
That is the whole point: the OTel-free network handler can read
:data:`TRACEPARENT_DEP_KEY` without making OpenTelemetry a hard dependency,
so tracing stays opt-in. It also sits below both ``middleware`` and
``network`` in the import graph, so either can import it "downward" without
coupling the two packages to each other.

The convention here is "centralise the shared, leave the single-use inline":
strings referenced from more than one module (or that form a closed
vocabulary consumers must match against) live here; attribute keys used at a
single site stay as literals next to their use.
"""

# ── Propagation contract ────────────────────────────────────────────────────
# Key under which the network handler relays an inbound envelope's W3C
# traceparent to the agent-side ``TelemetryMiddleware`` via
# ``context.dependencies`` (the ``Envelope`` itself never reaches middleware).
# Producer: autogen/beta/network/client/handlers.py
# Consumer: autogen/beta/middleware/builtin/telemetry.py
TRACEPARENT_DEP_KEY = "ag2.otel.traceparent"

# ── Tracer identity ─────────────────────────────────────────────────────────
# Shared so every AG2 tracer (agent middleware + hub) reports the same
# instrumentation scope and schema URL on the spans it emits.
OTEL_SCHEMA_URL = "https://opentelemetry.io/schemas/1.11.0"
OTEL_INSTRUMENTING_MODULE = "opentelemetry.instrumentation.ag2.beta"

# ── Span-type discriminator ─────────────────────────────────────────────────
# Attribute key + its closed value vocabulary. This is the field that
# consumers (the disk-JSONL reader, eval / query code) filter on, so the
# values are centralised even though each is set at a single producer site.
ATTR_SPAN_TYPE = "ag2.span.type"
SPAN_TYPE_AGENT = "agent"  # invoke_agent — middleware
SPAN_TYPE_LLM = "llm"  # chat — middleware
SPAN_TYPE_TOOL = "tool"  # execute_tool — middleware
SPAN_TYPE_HUMAN_INPUT = "human_input"  # await_human_input — middleware
SPAN_TYPE_ENVELOPE = "envelope"  # network.envelope — hub
SPAN_TYPE_CHANNEL = "channel"  # network.channel — hub
SPAN_TYPE_TASK = "task"  # network.task — hub
SPAN_TYPE_AGENT_LIFETIME = "agent_lifetime"  # agent.lifetime — hub
SPAN_TYPE_AGENT_EVENT = "agent_event"  # agent.resume_set / skill_set / rule_set — hub

# ── SpanLink kinds ──────────────────────────────────────────────────────────
# ``ag2.link.kind`` attribute on cross-trace ``Link``s. ``in_channel`` is
# emitted by both the envelope tracer and the listener, hence shared.
ATTR_LINK_KIND = "ag2.link.kind"
LINK_IN_CHANNEL = "in_channel"  # span happened in this channel
LINK_TRIGGERED_BY = "triggered_by"  # span was triggered by the caller's current span

# ── Network attribute keys (``ag2.network.*``) ──────────────────────────────
# Emitted by the envelope tracer and/or the HubTelemetryListener. The first
# four are emitted by both; the rest by a single site — but every key lives
# here so consumers (eval scorecards, the JSONL reader, TraceQL queries) have
# one importable source and producer code carries no ``ag2.*`` literals. This
# mirrors how ``hub/audit.py`` centralises its ``AUDIT_KIND_*`` vocabulary.
ATTR_NET_CHANNEL_ID = "ag2.network.channel_id"
ATTR_NET_SENDER_ID = "ag2.network.sender_id"
ATTR_NET_EVENT_TYPE = "ag2.network.event_type"
ATTR_NET_ENVELOPE_ID = "ag2.network.envelope_id"
ATTR_NET_CAPABILITY = "ag2.network.capability"
ATTR_NET_OUTCOME = "ag2.network.outcome"
ATTR_NET_TASK_ID = "ag2.network.task_id"
ATTR_NET_OWNER_ID = "ag2.network.owner_id"
ATTR_NET_RECIPIENT_ID = "ag2.network.recipient_id"
ATTR_NET_DISPATCH_FAILURES = "ag2.network.dispatch_failures"
ATTR_NET_MANIFEST_TYPE = "ag2.network.manifest_type"
ATTR_NET_CREATOR_ID = "ag2.network.creator_id"
ATTR_NET_CAUSATION_ID = "ag2.network.causation_id"
ATTR_NET_AUDIENCE = "ag2.network.audience"

# ── Agent attribute keys (``ag2.agent.*``) ──────────────────────────────────
# The ``resume_source`` *values* are not defined here — they reuse
# ``RESUME_SOURCE_TENANT`` / ``RESUME_SOURCE_OBSERVED`` from
# ``autogen.beta.network.hub.audit`` so the span value matches the audit record.
ATTR_AGENT_ID = "ag2.agent.id"
ATTR_AGENT_CAPABILITY = "ag2.agent.capability"
ATTR_AGENT_OUTCOME = "ag2.agent.outcome"
ATTR_AGENT_RESUME_SOURCE = "ag2.agent.resume_source"
ATTR_AGENT_SKILL_REMOVED = "ag2.agent.skill_removed"

# ── Diagnostic span-event attribute keys ────────────────────────────────────
# Failure-path breadcrumbs read in a trace viewer (error / expectation /
# inbox-pressure events) and human-input prompt/response capture.
ATTR_ERROR_TYPE = "ag2.error.type"
ATTR_ERROR_MESSAGE = "ag2.error.message"
ATTR_EXPECTATION_NAME = "ag2.expectation.name"
ATTR_EXPECTATION_ON_VIOLATION = "ag2.expectation.on_violation"
ATTR_EXPECTATION_VIOLATORS = "ag2.expectation.violators"
ATTR_INBOX_PENDING = "ag2.inbox.pending"
ATTR_INBOX_CAP = "ag2.inbox.cap"

# ── Human-input capture ─────────────────────────────────────────────────────
# Prompt/response text captured on ``human_input`` spans, read back by the
# eval trace reconstructor to rebuild HITL turns.
ATTR_HUMAN_INPUT_PROMPT = "ag2.human_input.prompt"
ATTR_HUMAN_INPUT_RESPONSE = "ag2.human_input.response"
