# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Prebuilt scorers — frozen versions of the most common scorer patterns.

Each name below is a *factory* that returns a :class:`Scorer`. Drop
them straight into ``scorers=[...]``::

    from ag2.eval.scorers import (
        tool_called,
        no_tool_errors,
        final_answer_matches,
        token_budget,
    )

    scorers = [
        tool_called("get_weather"),
        no_tool_errors(),
        final_answer_matches(field="city", matcher="contains"),
        token_budget(2_000),
    ]

The four deterministic prebuilts above are a deliberately small starter
catalog. ``agent_judge`` adds LLM grading: a *single-purpose* Agent-as-judge
(one criterion → one ``Feedback`` key). Compose several for a multi-dimensional
scorecard::

    from ag2.eval.scorers import agent_judge

    scorers = [
        agent_judge(config, criterion="Answer is correct vs reference.", key="correctness"),
        agent_judge(config, criterion="Claims are grounded in tool results.", key="faithfulness"),
    ]

``memory_recall`` / ``memory_precision`` / ``stale_fact_injection`` grade the
memory a run injected (see :mod:`.memory`); ``build_memory_suite`` and
``evaluate_retriever`` measure a retriever offline, without any agent run.
"""

from .attribution import ERROR_MODES, Attribution, failure_attribution
from .correctness import final_answer_matches
from .cost import token_budget
from .human_pairwise import export_pairwise_cases, human_labels, human_pairwise
from .judge import Verdict, agent_judge
from .memory import (
    MemoryEvalCase,
    MemoryEvalReport,
    Retriever,
    build_memory_suite,
    evaluate_retriever,
    memory_precision,
    memory_recall,
    stale_fact_injection,
)
from .pairwise_judge import PairwiseVerdict, pairwise_judge
from .threshold import threshold
from .tools import no_tool_errors, tool_called

__all__ = (
    "ERROR_MODES",
    "Attribution",
    "MemoryEvalCase",
    "MemoryEvalReport",
    "PairwiseVerdict",
    "Retriever",
    "Verdict",
    "agent_judge",
    "build_memory_suite",
    "evaluate_retriever",
    "export_pairwise_cases",
    "failure_attribution",
    "final_answer_matches",
    "human_labels",
    "human_pairwise",
    "memory_precision",
    "memory_recall",
    "no_tool_errors",
    "pairwise_judge",
    "stale_fact_injection",
    "threshold",
    "token_budget",
    "tool_called",
)
