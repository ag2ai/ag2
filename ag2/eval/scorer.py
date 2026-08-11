# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Scorer — the unit of grading inside the eval framework.

A scorer is a function that grades one task's :class:`Trace` against the
task definition. Authors decorate plain Python functions with
:func:`scorer`; the decorator introspects the function's parameters and,
at call time, injects only what the function asked for, by name. This
matches LangSmith's settled shape::

    @scorer
    def called_get_weather(trace):
        return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1


    @scorer
    def city_argument_correct(trace, reference_outputs):
        calls = trace.events_of(ToolCallEvent, name="get_weather")
        if not calls:
            return False
        return calls[0].arguments.get("city") == reference_outputs["city"]

Scorers can be sync or async — the decorator handles both. They should
be pure functions of their inputs (no I/O, no global state).

Return values are normalized into one or more :class:`Feedback` records:

* ``bool`` / ``int`` / ``float`` → ``Feedback(key=<fn name>, score=<value>)``
* ``str`` → ``Feedback(key=<fn name>, value=<value>)`` (categorical)
* :class:`Feedback` → used directly
* ``list[Feedback]`` → multiple records from one call
* ``None`` → skipped (no record produced)
* anything else → :class:`ScorerReturnTypeError`

A scorer that *raises* an exception does not fail the run. The runner
catches it, records a :class:`Feedback` with ``score=None`` and an
explanatory comment, logs a warning, and moves on.
"""

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

from ._types import Feedback, ScorerReturnTypeError
from .dataset import Task
from .trace import Trace

__all__ = (
    "Scorer",
    "ScorerFn",
    "scorer",
)


logger = logging.getLogger(__name__)


ScorerFn: TypeAlias = Callable[..., Any] | Callable[..., Awaitable[Any]]
"""Any callable a user can pass to :func:`scorer`. Signature is introspected at decoration time."""


_INJECTABLE_PARAMS: frozenset[str] = frozenset({
    "inputs",
    "outputs",
    "reference_outputs",
    "trace",
    "task",
})


class Scorer:
    """Wraps a user function as a callable scoring unit.

    Most users will not construct this directly — they write a function
    and decorate it with :func:`scorer`. Prebuilt scorer factories (e.g.
    ``tool_called(name)``) construct :class:`Scorer` instances
    programmatically to supply a meaningful ``key`` independent of any
    closure's function name.

    Args:
        fn: The scoring function. May be sync or async. May declare any
            subset of the injectable parameters (``inputs``, ``outputs``,
            ``reference_outputs``, ``trace``, ``task``); anything else
            raises ``TypeError`` at construction time.
        key: Stable identifier for the feedback this scorer produces.
            Defaults to ``fn.__name__``. Pass-rate and stats lookups on
            :class:`~ag2.eval.RunResult` use this key.
    """

    __slots__ = ("_fn", "_key", "_params", "_is_async")

    def __init__(
        self,
        fn: ScorerFn,
        *,
        key: str | None = None,
    ) -> None:
        self._fn = fn
        self._key = key if key is not None else getattr(fn, "__name__", "scorer")
        self._params = _validate_signature(fn, self._key)
        self._is_async = inspect.iscoroutinefunction(fn)

    @property
    def key(self) -> str:
        """The feedback key this scorer emits (defaults to the wrapped function's name)."""
        return self._key

    async def __call__(
        self,
        *,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any] | None,
        trace: Trace,
        task: Task,
    ) -> list[Feedback]:
        """Run the scorer for one task and return zero or more :class:`Feedback` records.

        Only the parameters the wrapped function declared are passed
        through. Exceptions raised by the wrapped function are captured
        as a ``Feedback(score=None, comment=...)`` record and logged at
        WARNING; the runner never sees the exception.
        """
        available: dict[str, Any] = {
            "inputs": inputs,
            "outputs": outputs,
            "reference_outputs": reference_outputs,
            "trace": trace,
            "task": task,
        }
        call_args = {name: available[name] for name in self._params}

        try:
            if self._is_async:
                result = await self._fn(**call_args)
            else:
                result = self._fn(**call_args)
        except Exception as exc:
            logger.warning("Scorer %r raised %s: %s", self._key, type(exc).__name__, exc)
            return [
                Feedback(
                    key=self._key,
                    score=None,
                    comment=f"scorer raised: {type(exc).__name__}: {exc}",
                )
            ]

        return self._normalize(result)

    def _normalize(self, result: Any) -> list[Feedback]:
        """Normalize a scorer's return value into a list of :class:`Feedback`.

        Order matters: ``bool`` is a subclass of ``int``, so it must be
        tested first; otherwise ``True`` would never produce a boolean
        feedback. ``Feedback`` is checked before ``list`` because a
        future :class:`Feedback` subclass should still be treated as a
        single record.
        """
        if result is None:
            return []
        if isinstance(result, Feedback):
            return [result]
        if isinstance(result, list):
            for index, item in enumerate(result):
                if not isinstance(item, Feedback):
                    raise ScorerReturnTypeError(
                        f"scorer {self._key!r} returned list[...] with non-Feedback at index "
                        f"{index}: got {type(item).__name__}"
                    )
            return list(result)
        if isinstance(result, bool):
            return [Feedback(key=self._key, score=result)]
        if isinstance(result, (int, float)):
            return [Feedback(key=self._key, score=result)]
        if isinstance(result, str):
            return [Feedback(key=self._key, value=result)]
        raise ScorerReturnTypeError(
            f"scorer {self._key!r} returned unsupported type {type(result).__name__}; "
            f"expected bool, int, float, str, Feedback, list[Feedback], or None"
        )


def scorer(fn: ScorerFn) -> Scorer:
    """Decorate a function as a scorer.

    Example::

        @scorer
        def called_get_weather(trace):
            return len(trace.events_of(ToolCallEvent, name="get_weather")) == 1

    The decorated object is a :class:`Scorer` instance. Pass it directly
    to :func:`ag2.eval.run_agent` via the ``scorers=`` argument.
    """
    return Scorer(fn)


def _validate_signature(fn: ScorerFn, key: str) -> tuple[str, ...]:
    """Return the names of the parameters ``fn`` declares, in declaration order.

    Raises ``TypeError`` for unsupported parameter shapes:

    * ``*args`` or ``**kwargs`` — the decorator only injects by name, so
      variadic params are ambiguous.
    * Positional-only params — we always call with kwargs.
    * Any parameter outside the injectable set
      (``inputs`` / ``outputs`` / ``reference_outputs`` / ``trace`` / ``task``).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"scorer {key!r}: could not introspect signature: {exc}") from exc

    params: list[str] = []
    for name, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            raise TypeError(f"scorer {key!r}: *{name} is not supported; declare named parameters instead")
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            raise TypeError(f"scorer {key!r}: **{name} is not supported; declare named parameters instead")
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(
                f"scorer {key!r}: parameter {name!r} is positional-only; "
                f"use a regular parameter so it can be injected by name"
            )
        if name not in _INJECTABLE_PARAMS:
            allowed = ", ".join(sorted(_INJECTABLE_PARAMS))
            raise TypeError(f"scorer {key!r}: parameter {name!r} is not injectable; allowed parameters are: {allowed}")
        params.append(name)
    return tuple(params)
