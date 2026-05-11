# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Trace — a read-only view of one captured eval run.

A :class:`Trace` is what every scorer sees: the full ordered sequence of
events the agent emitted, the final :class:`~autogen.beta.AgentReply` (or
exception), and wall-clock duration. Trace is the AG2-native equivalent
of LangSmith's ``Run`` object — but the events are *typed*
(``ToolCallEvent``, ``ToolResultEvent``, ``HaltEvent``, …), so scorers
consume structure directly instead of parsing free-form text.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeVar

from autogen.beta.agent import AgentReply
from autogen.beta.events import BaseEvent, HaltEvent, ModelResponse

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
    by event type, and :attr:`tokens` / :attr:`duration_ms` / :attr:`halted`
    / :attr:`exception` for run-level signals.

    Trace is constructed by the eval runner and has no equivalent inside
    ``autogen.beta`` itself — it is an eval-only view object.
    """

    __slots__ = ("_events", "_reply", "_exception", "_duration_ms")

    def __init__(
        self,
        *,
        events: Iterable[BaseEvent],
        reply: AgentReply[Any, Any] | None,
        exception: BaseException | None,
        duration_ms: int,
    ) -> None:
        self._events: tuple[BaseEvent, ...] = tuple(events)
        self._reply = reply
        self._exception = exception
        self._duration_ms = duration_ms

    @property
    def events(self) -> tuple[BaseEvent, ...]:
        """Every event emitted on the agent's stream during this task, in order."""
        return self._events

    @property
    def reply(self) -> AgentReply[Any, Any] | None:
        """Final ``AgentReply`` from ``agent.ask(...)``, or ``None`` if the run raised."""
        return self._reply

    @property
    def exception(self) -> BaseException | None:
        """Exception raised by ``agent.ask(...)``, or ``None`` on clean completion."""
        return self._exception

    @property
    def halted(self) -> bool:
        """``True`` iff a :class:`HaltEvent` was emitted (e.g. by an ``AlertPolicy``)."""
        return any(isinstance(e, HaltEvent) for e in self._events)

    @property
    def duration_ms(self) -> int:
        """Wall-clock duration of the task, in milliseconds."""
        return self._duration_ms

    @property
    def tokens(self) -> TokenUsage:
        """Token counts summed across every :class:`ModelResponse` in this run."""
        input_total = 0
        output_total = 0
        cache_creation = 0
        cache_read = 0
        for event in self._events:
            if not isinstance(event, ModelResponse):
                continue
            usage = event.usage
            input_total += int(usage.prompt_tokens or 0)
            output_total += int(usage.completion_tokens or 0)
            cache_creation += int(usage.cache_creation_input_tokens or 0)
            cache_read += int(usage.cache_read_input_tokens or 0)
        return TokenUsage(
            input=input_total,
            output=output_total,
            cache_creation=cache_creation,
            cache_read=cache_read,
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
