# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""View policies — per-participant projection from WAL to ModelEvents.

A participant's effective LLM context for a turn is
``[layer_C_summary, *layer_B_projection, current_envelope]``. Layer B
is what view policies produce.

Built-ins: ``FullTranscript`` (verbatim) and ``WindowedSummary``
(bounded tail + head summary, composes with framework-core
``compact.py``). Envelope rendering is supplied per-call by the
channel's adapter via ``ChannelAdapter.render_envelope``. Sender
identity is supplied per-call by the handler via a ``NameResolver``
sourced from the hub's passport directory.
"""

from .base import EnvelopeRenderer, NameResolver, ViewPolicy, default_name_resolver
from .builtin import FullTranscript, NamedTranscript, NamedWindowedSummary, WindowedSummary

__all__ = (
    "EnvelopeRenderer",
    "FullTranscript",
    "NameResolver",
    "NamedTranscript",
    "NamedWindowedSummary",
    "ViewPolicy",
    "WindowedSummary",
    "default_name_resolver",
)
