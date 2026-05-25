# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Eval target Protocol — what the runner drives end-to-end on one task.

A :class:`EvalTarget` is anything the runner can call ``.ask(prompt, ...)``
on. In v0 the canonical implementation is :class:`~autogen.beta.Agent`,
which satisfies the Protocol structurally — its ``ask`` accepts a string
as the first positional argument and the runner's keyword arguments
(``stream``, ``observers``).

The Protocol exists so v0 single-agent eval and v1 multi-agent / network
eval share one runner contract. Wrapper targets (multi-turn drivers via
``reply.ask``), :class:`~autogen.beta.Agent` itself, and future
``NetworkTarget`` / ``WorkflowTarget`` / ``DiscussionTarget``
implementations all satisfy the same shape; the runner doesn't change.
"""

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from autogen.beta.observers import Observer
from autogen.beta.stream import Stream

__all__ = ("EvalTarget",)


@runtime_checkable
class EvalTarget(Protocol):
    """Something the eval runner can drive end-to-end on one task.

    The runner calls ``target.ask(prompt, stream=..., observers=[capture])``
    exactly once per task. Events emitted on ``stream`` during that call
    become the task's :class:`~autogen.beta.eval.Trace`. The return value
    (typically an :class:`~autogen.beta.AgentReply`-like object) is
    duck-typed for ``body`` and ``response`` attributes when normalizing
    scorer ``outputs``.

    Implementations in v0:

    * :class:`~autogen.beta.Agent` — the canonical single-agent target.
    * Lightweight wrappers around an :class:`~autogen.beta.Agent` that
      drive ``reply.ask`` continuations internally (multi-turn pattern).

    Expected in v1+:

    * ``NetworkTarget`` — multi-agent flows via the network hub.
    * ``WorkflowTarget`` / ``DiscussionTarget`` — adapter-specific
      orchestrations that surface as a single task to the runner.

    All v0 and v1 implementations grade against the same
    :class:`~autogen.beta.eval.Trace` and :class:`~autogen.beta.eval.RunResult`
    machinery; the only thing that varies by target type is *which*
    stream(s) the runner subscribes to and how the target is driven.
    """

    async def ask(
        self,
        prompt: str,
        /,
        *,
        stream: Stream | None = ...,
        observers: Iterable[Observer] = ...,
    ) -> Any: ...
