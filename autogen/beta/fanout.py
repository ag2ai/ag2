# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""fanout — run agent invocations in parallel.

:func:`fanout` is the go-to primitive for running multiple ``agent.ask()``
calls concurrently — whether against the same agent with different prompts,
different agents with the same prompt, or any mix.

Example::

    from autogen.beta import Agent, fanout
    from autogen.beta.config import OpenAIConfig

    agent = Agent("assistant", config=OpenAIConfig("gpt-4o-mini"))

    # Same agent, multiple prompts — returns list in input order
    replies = await fanout(agent, ["Summarize Paris.", "Summarize Rome."])
    for r in replies:
        print(r.body)

    # Multiple agents, same prompt (ensemble)
    replies = await fanout([formal, casual, technical], "Explain recursion.")

    # Heterogeneous pairs
    replies = await fanout([
        (agent_a, "Summarize document A."),
        (agent_b, "Summarize document B."),
    ])

    # Limit to 3 concurrent calls at most
    replies = await fanout(agent, prompts, max_concurrent=3)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from .agent import Agent, AgentReply

__all__ = ("fanout",)

# A job is a (agent, prompt) pair.
_Job = tuple["Agent[Any]", str]


@overload
async def fanout(
    agent: Agent[Any],
    prompts: list[str],
    *,
    max_concurrent: int | None = None,
) -> list[AgentReply[Any, Any]]: ...


@overload
async def fanout(
    agents: list[Agent[Any]],
    prompt: str,
    *,
    max_concurrent: int | None = None,
) -> list[AgentReply[Any, Any]]: ...


@overload
async def fanout(
    jobs: list[tuple[Agent[Any], str]],
    *,
    max_concurrent: int | None = None,
) -> list[AgentReply[Any, Any]]: ...


async def fanout(
    agent_or_agents_or_jobs: Agent[Any] | list[Agent[Any]] | list[tuple[Agent[Any], str]],
    prompt_or_prompts: str | list[str] | None = None,
    *,
    max_concurrent: int | None = None,
) -> list[AgentReply[Any, Any]]:
    """Run multiple ``agent.ask()`` calls concurrently and return results in order.

    Three calling conventions are supported:

    .. code-block:: python

        # 1. Same agent, many prompts
        replies = await fanout(agent, ["Summarize Paris.", "Summarize Rome."])

        # 2. Many agents, same prompt (ensemble / redundant)
        replies = await fanout([agent_a, agent_b, agent_c], "Explain recursion.")

        # 3. Explicit (agent, prompt) pairs
        replies = await fanout([
            (agent_a, "Prompt for A."),
            (agent_b, "Prompt for B."),
        ])

    Args:
        agent_or_agents_or_jobs:
            * A single :class:`~autogen.beta.Agent` (paired with *prompt_or_prompts* as a list).
            * A list of :class:`~autogen.beta.Agent` objects (all receive the same *prompt_or_prompts* string).
            * A list of ``(agent, prompt)`` pairs (no second argument needed).
        prompt_or_prompts:
            A single prompt string (for the list-of-agents form) or a list of prompt strings
            (for the single-agent form).  Omit when passing explicit ``(agent, prompt)`` pairs.
        max_concurrent:
            Maximum number of concurrent ``agent.ask()`` calls.  ``None`` (default) means all
            calls run at once.  Set this to avoid overwhelming rate limits or shared resources
            when the job list is large.

    Returns:
        A list of :class:`~autogen.beta.AgentReply` objects, one per job, in the **same order**
        as the input jobs.

    Raises:
        ValueError: If the arguments don't match any of the supported calling conventions.
        Exception: Any exception raised by ``agent.ask()`` propagates directly.

    Examples::

        # With concurrency cap
        replies = await fanout(agent, long_list_of_prompts, max_concurrent=5)

        # Inspect results
        for reply in replies:
            print(reply.body)
            print(reply.usage.total_tokens)
    """
    jobs = _normalize_jobs(agent_or_agents_or_jobs, prompt_or_prompts)

    if not jobs:
        return []

    if max_concurrent is None:
        return list(await asyncio.gather(*[agent.ask(prompt) for agent, prompt in jobs]))

    sem = asyncio.Semaphore(max_concurrent)

    async def _run(agent: Agent[Any], prompt: str) -> AgentReply[Any, Any]:
        async with sem:
            return await agent.ask(prompt)

    return list(await asyncio.gather(*[_run(agent, prompt) for agent, prompt in jobs]))


def _normalize_jobs(
    first: Agent[Any] | list[Agent[Any]] | list[tuple[Agent[Any], str]],
    second: str | list[str] | None,
) -> list[tuple[Agent[Any], str]]:
    """Resolve the three calling conventions into a flat list of (agent, prompt) pairs."""
    from .agent import Agent  # local import to avoid circular dependency

    if isinstance(first, Agent):
        # Convention 1: single agent + list of prompts
        if not isinstance(second, list):
            raise ValueError("When the first argument is an Agent, the second argument must be a list[str] of prompts.")
        return [(first, p) for p in second]

    if not first:
        return []

    first_item = first[0]

    if isinstance(first_item, Agent):
        # Convention 2: list of agents + single shared prompt
        if not isinstance(second, str):
            raise ValueError(
                "When the first argument is a list[Agent], the second argument must be a single prompt string."
            )
        agents: list[Agent] = first  # type: ignore[assignment]
        return [(a, second) for a in agents]

    # Convention 3: list of (agent, prompt) pairs
    if second is not None:
        raise ValueError("When the first argument is a list of (agent, prompt) pairs, no second argument is expected.")
    return list(first)  # type: ignore[arg-type]
