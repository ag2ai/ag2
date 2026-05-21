# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``HubTelemetryListener`` — OpenTelemetry tracing as a :class:`HubListener`.

Emits OTel spans for hub state transitions, mirroring the
:class:`AuditLog` pattern. Each entity is its own bounded trace:

* ``network.channel {type}`` — long-lived, open from ``created`` to
  ``closed`` / ``expired``. Channel-scoped signal events (expectation
  fires, rejections, dispatch / turn failures) attach here.
* ``agent.lifetime {name}`` — long-lived, open from ``registered`` to
  ``unregistered``. Identity-change spans (``agent.resume_set`` /
  ``skill_set`` / ``rule_set``) nest under it; inbox-pressure events
  attach here.
* ``network.task {capability}`` — single-shot at the terminal event.
  ``started`` / ``progress`` never reach listeners, so the span is
  created and ended in one call with ``start_time`` backdated to the
  task's ``started_at`` (added to the terminal payload by the hub).

Spans are exported via the shared ``TracerProvider`` (collector) and
mirrored to ``/telemetry/spans.jsonl`` on the hub's ``KnowledgeStore``.
The Hub itself owns ``network.envelope`` spans (see ``_envelope_tracing``)
— this listener does not create them, only annotates via the
channel span.

Opt-in: register explicitly with ``hub.register_listener(...)``. The Hub
does not install it automatically.
"""

import contextlib
import logging
from collections.abc import Awaitable, Callable

try:
    from opentelemetry.trace import Link, SpanContext, SpanKind, Status, StatusCode, set_span_in_context

    from ._envelope_tracing import get_tracer, iso_to_ns, serialize_span
except ImportError as _err:  # pragma: no cover - exercised via packaging
    raise ImportError(
        "OpenTelemetry packages are required for HubTelemetryListener. Install them with: pip install ag2[tracing]"
    ) from _err

from autogen.beta._telemetry_consts import (
    ATTR_AGENT_CAPABILITY,
    ATTR_AGENT_ID,
    ATTR_AGENT_OUTCOME,
    ATTR_AGENT_RESUME_SOURCE,
    ATTR_AGENT_SKILL_REMOVED,
    ATTR_ERROR_MESSAGE,
    ATTR_ERROR_TYPE,
    ATTR_EXPECTATION_NAME,
    ATTR_EXPECTATION_ON_VIOLATION,
    ATTR_EXPECTATION_VIOLATORS,
    ATTR_INBOX_CAP,
    ATTR_INBOX_PENDING,
    ATTR_LINK_KIND,
    ATTR_NET_CAPABILITY,
    ATTR_NET_CHANNEL_ID,
    ATTR_NET_CREATOR_ID,
    ATTR_NET_ENVELOPE_ID,
    ATTR_NET_EVENT_TYPE,
    ATTR_NET_MANIFEST_TYPE,
    ATTR_NET_OUTCOME,
    ATTR_NET_OWNER_ID,
    ATTR_NET_RECIPIENT_ID,
    ATTR_NET_SENDER_ID,
    ATTR_NET_TASK_ID,
    ATTR_SPAN_TYPE,
    LINK_IN_CHANNEL,
    SPAN_TYPE_AGENT_EVENT,
    SPAN_TYPE_AGENT_LIFETIME,
    SPAN_TYPE_CHANNEL,
    SPAN_TYPE_TASK,
)
from autogen.beta.knowledge import KnowledgeStore

from .audit import RESUME_SOURCE_OBSERVED, RESUME_SOURCE_TENANT
from .layout import spans_path
from .listener import BaseHubListener

logger = logging.getLogger(__name__)

SpanRecordSubscriber = Callable[[dict], Awaitable[None]]

__all__ = ("HubTelemetryListener", "SpanRecordSubscriber")

_TERMINAL_TASK_KINDS = ("completed", "failed", "expired", "mirror_failed")


class HubTelemetryListener(BaseHubListener):
    """Translate hub state transitions into OpenTelemetry spans.

    Args:
        store: The hub's ``KnowledgeStore``. Span records are appended to
            ``/telemetry/spans.jsonl`` as one JSON object per line.
        tracer_provider: Optional ``TracerProvider``. Defaults to the
            global provider so it shares the resource (and sampler) with
            agent-side ``TelemetryMiddleware``.
        span_attributes: Extra key/value pairs stamped on every span this
            listener emits (e.g. ``{"ag2.org.id": "..."}``).
    """

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        tracer_provider: object | None = None,
        span_attributes: dict[str, str] | None = None,
    ) -> None:
        self._store = store
        self._tracer = get_tracer(tracer_provider)
        self._stamp = dict(span_attributes or {})
        self._channel_spans: dict[str, object] = {}
        self._agent_spans: dict[str, object] = {}
        self._subscribers: list[SpanRecordSubscriber] = []
        self._bytes_written = 0

    # ── Public surface ────────────────────────────────────────────────────────

    @property
    def bytes_written(self) -> int:
        """Process-local byte counter, surfaced via ``Hub.health()``."""
        return self._bytes_written

    def subscribe(self, callback: SpanRecordSubscriber) -> None:
        """Attach a live callback fired with each span record as it is written."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: SpanRecordSubscriber) -> None:
        """Detach a previously-registered subscriber. No-op if absent."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(callback)

    def channel_span_context(self, channel_id: str) -> "SpanContext | None":
        """The ``SpanContext`` of an open channel span, for envelope→channel SpanLinks.

        Called by the Hub when starting a ``network.envelope`` span so it
        can link back to the channel. Returns ``None`` if the channel
        span isn't open (no listener saw ``created``, or hub restart).
        """
        span = self._channel_spans.get(channel_id)
        return span.get_span_context() if span is not None else None

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _emit(self, span: object) -> None:
        """Serialise an ended span, append it to disk, notify subscribers."""
        record, line = serialize_span(span)
        await self._store.append(spans_path(), line)
        self._bytes_written += len(line.encode("utf-8"))
        for subscriber in self._subscribers:
            try:
                await subscriber(record)
            except Exception:
                logger.exception("telemetry subscriber raised: span=%s", record.get("name"))

    def _channel_link(self, channel_id: str) -> "list[Link]":
        ctx = self.channel_span_context(channel_id)
        if ctx is None or not ctx.is_valid:
            return []
        return [Link(ctx, attributes={ATTR_LINK_KIND: LINK_IN_CHANNEL})]

    # ── Agent lifecycle ───────────────────────────────────────────────────────

    async def on_agent_event(self, agent_id: str, kind: str, payload: dict) -> None:
        if kind == "registered":
            passport = payload.get("passport")
            name = getattr(passport, "name", None) or agent_id
            span = self._tracer.start_span(
                f"agent.lifetime {name}",
                kind=SpanKind.INTERNAL,
                attributes={
                    **self._stamp,
                    ATTR_SPAN_TYPE: SPAN_TYPE_AGENT_LIFETIME,
                    ATTR_AGENT_ID: agent_id,
                    "gen_ai.agent.name": name,
                },
            )
            self._agent_spans[agent_id] = span
            return

        if kind == "unregistered":
            span = self._agent_spans.pop(agent_id, None)
            if span is None:
                logger.warning("telemetry: unregistered for unknown agent_id=%s", agent_id)
                return
            span.add_event("unregistered", attributes={"name": payload.get("name") or agent_id})
            span.end()
            await self._emit(span)
            return

        # Identity changes → single-shot child spans under agent.lifetime.
        parent = self._agent_spans.get(agent_id)
        if parent is None:
            logger.warning("telemetry: %s for unknown agent_id=%s", kind, agent_id)
            return
        await self._emit_agent_child(agent_id, kind, payload, parent)

    async def _emit_agent_child(self, agent_id: str, kind: str, payload: dict, parent: object) -> None:
        attributes: dict = {
            **self._stamp,
            ATTR_SPAN_TYPE: SPAN_TYPE_AGENT_EVENT,
            ATTR_AGENT_ID: agent_id,
        }
        if kind == "resume_set":
            name = "agent.resume_set"
            attributes[ATTR_AGENT_RESUME_SOURCE] = RESUME_SOURCE_TENANT
        elif kind == "observation_recorded":
            name = "agent.resume_set"
            attributes[ATTR_AGENT_RESUME_SOURCE] = RESUME_SOURCE_OBSERVED
            if payload.get("capability") is not None:
                attributes[ATTR_AGENT_CAPABILITY] = payload["capability"]
            if payload.get("outcome") is not None:
                attributes[ATTR_AGENT_OUTCOME] = str(payload["outcome"])
        elif kind == "skill_set":
            name = "agent.skill_set"
            attributes[ATTR_AGENT_SKILL_REMOVED] = bool(payload.get("removed", False))
        elif kind == "rule_set":
            name = "agent.rule_set"
        else:
            return  # unknown kind — ignore

        span = self._tracer.start_span(
            f"{name} {agent_id}",
            kind=SpanKind.INTERNAL,
            context=set_span_in_context(parent),
            attributes=attributes,
        )
        span.end()
        await self._emit(span)

    # ── Channel lifecycle ─────────────────────────────────────────────────────

    async def on_channel_event(self, channel_id: str, kind: str, payload: dict) -> None:
        if kind == "created":
            metadata = payload.get("metadata")
            attributes: dict = {
                **self._stamp,
                ATTR_SPAN_TYPE: SPAN_TYPE_CHANNEL,
                ATTR_NET_CHANNEL_ID: channel_id,
            }
            if metadata is not None:
                attributes[ATTR_NET_MANIFEST_TYPE] = metadata.manifest.type
                attributes[ATTR_NET_CREATOR_ID] = metadata.creator_id
            manifest_type = metadata.manifest.type if metadata is not None else "channel"
            span = self._tracer.start_span(
                f"network.channel {manifest_type}",
                kind=SpanKind.INTERNAL,
                attributes=attributes,
            )
            self._channel_spans[channel_id] = span
            return

        span = self._channel_spans.get(channel_id)
        if span is None:
            logger.warning("telemetry: %s for unknown channel_id=%s", kind, channel_id)
            return

        if kind == "opened":
            span.add_event("opened")
        elif kind in ("closed", "expired"):
            span.add_event(kind, attributes={"reason": str(payload.get("reason") or "")})
            if kind == "expired":
                span.set_status(Status(StatusCode.ERROR, "expired"))
            self._channel_spans.pop(channel_id, None)
            span.end()
            await self._emit(span)

    # ── Channel-attached signal events ────────────────────────────────────────

    async def on_expectation_fired(self, channel_id: str, expectation: object, violation: object) -> None:
        span = self._channel_spans.get(channel_id)
        if span is None:
            logger.warning("telemetry: expectation fired for unknown channel_id=%s", channel_id)
            return
        exp = violation.expectation
        span.add_event(
            f"expectation.{exp.name}",
            attributes={
                ATTR_EXPECTATION_NAME: exp.name,
                ATTR_EXPECTATION_ON_VIOLATION: str(exp.on_violation),
                ATTR_EXPECTATION_VIOLATORS: list(violation.violator_ids),
            },
        )

    async def on_envelope_rejected(self, envelope: object, reason: object) -> None:
        span = self._channel_spans.get(envelope.channel_id)
        if span is None:
            return
        span.add_event(
            "envelope_rejected",
            attributes={
                ATTR_NET_SENDER_ID: envelope.sender_id,
                ATTR_NET_EVENT_TYPE: envelope.event_type,
                ATTR_ERROR_TYPE: type(reason).__name__,
                ATTR_ERROR_MESSAGE: str(reason),
            },
        )

    async def on_dispatch_failed(self, envelope: object, recipient_id: str, reason: BaseException) -> None:
        span = self._channel_spans.get(envelope.channel_id)
        if span is None:
            return
        span.add_event(
            "dispatch_failed",
            attributes={
                ATTR_NET_RECIPIENT_ID: recipient_id,
                ATTR_ERROR_TYPE: type(reason).__name__,
                ATTR_ERROR_MESSAGE: str(reason),
            },
        )

    async def on_turn_failed(self, channel_id: str, agent_id: str, envelope_id: str, exc: BaseException) -> None:
        span = self._channel_spans.get(channel_id)
        if span is None:
            return
        span.add_event(
            "turn_failed",
            attributes={
                ATTR_AGENT_ID: agent_id,
                ATTR_NET_ENVELOPE_ID: envelope_id,
                ATTR_ERROR_TYPE: type(exc).__name__,
                ATTR_ERROR_MESSAGE: str(exc),
            },
        )

    async def on_inbox_pressure(self, agent_id: str, pending: int, cap: int) -> None:
        # Inbox pressure is agent-scoped (no channel_id), so it attaches
        # to the agent.lifetime span rather than a channel span.
        span = self._agent_spans.get(agent_id)
        if span is None:
            return
        span.add_event(
            "inbox_pressure",
            attributes={ATTR_INBOX_PENDING: int(pending), ATTR_INBOX_CAP: int(cap)},
        )

    # ── Task lifecycle (single-shot at terminal) ──────────────────────────────

    async def on_task_event(self, task_id: str, kind: str, payload: dict) -> None:
        if kind not in _TERMINAL_TASK_KINDS:
            return  # started / progress never fan out; nothing to do

        start_ns = iso_to_ns(payload.get("started_at"))
        end_ns = iso_to_ns(payload.get("at"))
        capability = payload.get("capability") or "task"
        channel_id = payload.get("channel_id")

        attributes: dict = {
            **self._stamp,
            ATTR_SPAN_TYPE: SPAN_TYPE_TASK,
            ATTR_NET_TASK_ID: task_id,
            ATTR_NET_CAPABILITY: capability,
            ATTR_NET_OUTCOME: str(payload.get("outcome") or kind),
        }
        if channel_id:
            attributes[ATTR_NET_CHANNEL_ID] = channel_id
        if payload.get("owner_id"):
            attributes[ATTR_NET_OWNER_ID] = payload["owner_id"]

        span = self._tracer.start_span(
            f"network.task {capability}",
            kind=SpanKind.INTERNAL,
            start_time=start_ns,
            links=self._channel_link(channel_id) if channel_id else None,
            attributes=attributes,
        )
        if kind == "completed":
            span.set_status(Status(StatusCode.OK))
        else:
            span.set_status(Status(StatusCode.ERROR, str(payload.get("reason") or kind)))
        span.end(end_time=end_ns)
        await self._emit(span)
