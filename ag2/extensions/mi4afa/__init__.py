# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""mi4afa: white-box failure attribution for multi-agent runs, by probing model activations.

When a team of agents fails, Automated Failure Attribution asks *which step*
went wrong and *which agent* made it. Asking an LLM judge is unreliable, yet
the answer is often linearly decodable from an open-weight model's internal
activations while it reads the transcript. This extension turns that finding
into a tool: an :class:`ActivationExtractor` caches one activation per turn,
a :class:`LogisticProbe` trained on labelled failures (e.g. Who&When) scores
each turn, and :class:`ProbeAttributor` selects the best layer on validation
data and names the decisive step. :func:`probe_failure_attribution` exposes it
as an ``ag2.eval`` scorer, the white-box counterpart of
:func:`ag2.eval.scorers.failure_attribution`.

Based on research by Wendy Zheng.

Maintainer: Liang Wu (@wuliang211)
Docs: https://docs.ag2.ai/latest/docs/user-guide/extensions/mi4afa
"""

from ag2.exceptions import missing_additional_dependency

from .prompt import (
    DEFAULT_POSTFIX,
    DEFAULT_PREFIX,
    DEFAULT_SYSTEM_PROMPT,
    PromptEncoding,
    PromptTemplate,
    TurnPositionError,
    build_prompt,
)
from .trace import TraceConversation, conversation_from_trace
from .types import ActivationSite, Component, Conversation, FitReport, SiteScore, Turn

_DEPENDENCIES = 'torch>=2.4,<3" "transformers>=4.56,<6'

try:
    from .activations import ActivationExtractor, ConversationActivations
    from .attributor import Example, ProbeAttributor
    from .probe import LogisticProbe
    from .scorer import probe_failure_attribution
except ImportError as e:
    ActivationExtractor = missing_additional_dependency("ActivationExtractor", _DEPENDENCIES, e)  # type: ignore[misc]
    ConversationActivations = missing_additional_dependency("ConversationActivations", _DEPENDENCIES, e)  # type: ignore[misc]
    Example = missing_additional_dependency("Example", _DEPENDENCIES, e)  # type: ignore[misc]
    LogisticProbe = missing_additional_dependency("LogisticProbe", _DEPENDENCIES, e)  # type: ignore[misc]
    ProbeAttributor = missing_additional_dependency("ProbeAttributor", _DEPENDENCIES, e)  # type: ignore[misc]
    probe_failure_attribution = missing_additional_dependency("probe_failure_attribution", _DEPENDENCIES, e)

__all__ = (
    "DEFAULT_POSTFIX",
    "DEFAULT_PREFIX",
    "DEFAULT_SYSTEM_PROMPT",
    "ActivationExtractor",
    "ActivationSite",
    "Component",
    "Conversation",
    "ConversationActivations",
    "Example",
    "FitReport",
    "LogisticProbe",
    "ProbeAttributor",
    "PromptEncoding",
    "PromptTemplate",
    "SiteScore",
    "TraceConversation",
    "Turn",
    "TurnPositionError",
    "build_prompt",
    "conversation_from_trace",
    "probe_failure_attribution",
)
