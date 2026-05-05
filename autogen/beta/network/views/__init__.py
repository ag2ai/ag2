# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""View policies — per-participant projection from WAL to ModelEvents.

A participant's effective LLM context for a turn is
``[layer_C_summary, *layer_B_projection, current_envelope]``. Layer B
is what view policies produce.

Built-ins: ``FullTranscript`` (verbatim) and ``WindowedSummary``
(bounded tail + head summary, composes with framework-core
``compact.py``).
"""

from .base import ViewPolicy
from .builtin import FullTranscript, WindowedSummary

__all__ = ("FullTranscript", "ViewPolicy", "WindowedSummary")
