# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""``TelemetryHubListener`` — OpenTelemetry spans for hub channel lifecycle.

Register via ``Hub.register_listener`` alongside any tenant listeners.
Emits instant (point-in-time) spans for:

* ``channel.opened``  — fired when the invite handshake completes
* ``channel.closed``  — fired on explicit close or invite failure
* ``channel.expired`` — fired when TTL sweeper expires a channel
* ``agent.turn_failed`` — fired when the notify handler raises

Attributes follow the ``ag2.*`` namespace used by
:class:`autogen.beta.middleware.builtin.telemetry.TelemetryMiddleware`;
both sets of spans appear under the same instrumentation scope so
they compose naturally in Jaeger / Zipkin / OTLP backends.

OpenTelemetry is an optional dependency. Import errors at construction
time propagate; no silent fall-through.
"""

from typing import TYPE_CHECKING

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError as _err:
    raise ImportError(
        "OpenTelemetry packages are required for TelemetryHubListener. Install them with: pip install ag2[tracing]"
    ) from _err

from .listener import BaseHubListener

if TYPE_CHECKING:
    from ..channel import ChannelMetadata

__all__ = ("TelemetryHubListener",)

_SCHEMA_URL = "https://opentelemetry.io/schemas/1.11.0"
_INSTRUMENTING_MODULE = "opentelemetry.instrumentation.ag2.beta"


def _get_tracer(tracer_provider: "TracerProvider | None" = None) -> "trace.Tracer":
    provider = tracer_provider or trace.get_tracer_provider()
    return provider.get_tracer(_INSTRUMENTING_MODULE, schema_url=_SCHEMA_URL)


class TelemetryHubListener(BaseHubListener):
    """Emit OpenTelemetry spans for hub channel lifecycle events.

    Instantaneous spans (start + immediately end) record *when* each
    lifecycle transition happened without keeping a long-lived span open
    for the channel's full duration (which could be hours).

    Args:
        tracer_provider: Optional TracerProvider. Defaults to the global provider.
    """

    def __init__(self, *, tracer_provider: "TracerProvider | None" = None) -> None:
        self._tracer = _get_tracer(tracer_provider)

    async def on_channel_event(
        self,
        channel_id: str,
        kind: str,
        payload: dict,
    ) -> None:
        metadata: ChannelMetadata | None = payload.get("metadata")

        if kind == "opened":
            with self._tracer.start_as_current_span(
                "channel.opened",
                kind=SpanKind.INTERNAL,
            ) as span:
                span.set_attribute("ag2.span.type", "channel")
                span.set_attribute("ag2.channel.id", channel_id)
                if metadata is not None:
                    span.set_attribute("ag2.channel.type", metadata.manifest.type)
                    span.set_attribute("ag2.channel.creator_id", metadata.creator_id)
                    span.set_attribute("ag2.channel.participant_count", len(metadata.participants))

        elif kind in ("closed", "expired"):
            with self._tracer.start_as_current_span(
                f"channel.{kind}",
                kind=SpanKind.INTERNAL,
            ) as span:
                span.set_attribute("ag2.span.type", "channel")
                span.set_attribute("ag2.channel.id", channel_id)
                reason = payload.get("reason") or ""
                if reason:
                    span.set_attribute("ag2.channel.close_reason", reason)
                if metadata is not None:
                    span.set_attribute("ag2.channel.type", metadata.manifest.type)
                    span.set_attribute("ag2.channel.creator_id", metadata.creator_id)

    async def on_turn_failed(
        self,
        channel_id: str,
        agent_id: str,
        envelope_id: str,
        exc: BaseException,
    ) -> None:
        with self._tracer.start_as_current_span(
            "agent.turn_failed",
            kind=SpanKind.INTERNAL,
        ) as span:
            span.set_attribute("ag2.span.type", "turn_failure")
            span.set_attribute("ag2.channel.id", channel_id)
            span.set_attribute("ag2.agent.id", agent_id)
            span.set_attribute("ag2.envelope_id", envelope_id)
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, str(exc))
