# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trace — a read-only view of one graded unit of work.

A :class:`Trace` is what every scorer sees: the ordered sequence of *typed*
events (``ModelResponse``, ``ToolCallEvent``, ``ToolResultEvent``, …) plus the
run-level signals — wall-clock duration and the exception, if the run crashed.
Scorers consume structure directly (``trace.events_of(ToolCallEvent)``) rather
than parsing free-form text.

A Trace is **reconstructed from OpenTelemetry spans** — the same path for a live
``run_agent`` and for grading stored / cloud traces via ``evaluate_traces`` — so it carries
exactly what those spans captured. Stream-only events that never become spans
(``HaltEvent``, ``ToolNotFoundEvent``) are absent on this path; see
:mod:`ag2.eval.sources._spans`.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypeVar

from ag2.events import BaseEvent
from ag2.usage import UsageReport

__all__ = (
    "TokenUsage",
    "Trace",
)


_E = TypeVar("_E", bound=BaseEvent)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts summed across every model call in one task.

    Cache tokens are reported separately from ``input`` / ``output`` so
    that scorers and dashboards can attribute prompt-cache savings
    without affecting raw input counts. :attr:`total` covers only
    input + output, matching what most provider price tables charge for.
    """

    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0

    @property
    def total(self) -> int:
        """Sum of ``input`` and ``output`` tokens (cache tokens excluded)."""
        return self.input + self.output


class Trace:
    """Read-only view of one captured eval run.

    Scorers receive a Trace through the ``trace`` parameter (resolved by
    name by the ``@scorer`` decorator). Use :meth:`events_of` to filter
    by event type, and :attr:`tokens` / :attr:`duration_ms` / :attr:`exception`
    for run-level signals.

    Trace is constructed by the eval runner and has no equivalent inside
    ``ag2`` itself — it is an eval-only view object.
    """

    __slots__ = ("_events", "_exception", "_duration_ms")

    def __init__(
        self,
        *,
        events: Iterable[BaseEvent],
        exception: BaseException | None,
        duration_ms: int,
    ) -> None:
        self._events: tuple[BaseEvent, ...] = tuple(events)
        self._exception = exception
        self._duration_ms = duration_ms

    @property
    def events(self) -> tuple[BaseEvent, ...]:
        """Every event emitted on the agent's stream during this task, in order."""
        return self._events

    @property
    def exception(self) -> BaseException | None:
        """Exception raised by ``agent.ask(...)``, or ``None`` on clean completion."""
        return self._exception

    @property
    def duration_ms(self) -> int:
        """Wall-clock duration of the task, in milliseconds."""
        return self._duration_ms

    @property
    def tokens(self) -> TokenUsage:
        """Token counts summed across every :class:`UsageEvent` in this run.

        ``UsageEvent`` is the framework's accounting record, emitted wherever
        tokens are spent, and :class:`~ag2.UsageReport` — the same aggregation
        the framework reports as your bill — is reused here rather than
        maintaining a parallel summation.

        Deliberately *not* :class:`ModelResponse`. Most spend never produces
        one: a delegating agent's workers report through a ``"subtask"`` rollup,
        history compaction and memory aggregation report usage alone, and live
        sessions map realtime usage straight to the accounting event. Counting
        responses *as well* would double every main-loop call, which emits both.
        """
        total = UsageReport.from_events(self._events).total
        return TokenUsage(
            input=int(total.prompt_tokens or 0),
            output=int(total.completion_tokens or 0),
            cache_creation=int(total.cache_creation_input_tokens or 0),
            cache_read=int(total.cache_read_input_tokens or 0),
        )

    def events_of(
        self,
        event_type: type[_E],
        *,
        name: str | None = None,
    ) -> tuple[_E, ...]:
        """Return events matching ``event_type`` (and optionally ``.name``).

        ``isinstance`` is used to test the type, so subclasses match too.
        When ``name`` is supplied, only events whose ``.name`` attribute
        equals it are returned — useful for tool events::

            trace.events_of(ToolCallEvent, name="get_weather")

        Events without a ``name`` attribute are excluded when ``name`` is set.

        Returns:
            A tuple preserving original event order.
        """
        if name is None:
            return tuple(e for e in self._events if isinstance(e, event_type))
        return tuple(e for e in self._events if isinstance(e, event_type) and getattr(e, "name", None) == name)
