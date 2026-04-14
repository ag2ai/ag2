# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Transforms pipeline — Phase 5a.1.

Runtime machinery for ``rule.transforms``. Every transform runs inside
the recipient's :class:`ActorClient`, never in the hub process. The
pipeline is rebuilt whenever the hub pushes a ``RuleChangedFrame``.

Public surface:

- :class:`Transform` — protocol, ``(envelope, ctx) -> envelope | None``.
- :class:`TransformContext` — per-invocation metadata (stage, client,
  session id, rule version, direction).
- :class:`TransformRegistry` — per-client named-transform registry.
- :class:`TransformPipeline` — the four-stage driver.
- :class:`NamedTransform` / :class:`PythonTransform` /
  :class:`HttpTransform` — the three MVP adapters.
- Error types: :class:`TransformError`,
  :class:`TransformLookupError`, :class:`TransformRejected`.
"""

from __future__ import annotations

from .adapters import HttpTransform, NamedTransform, PythonTransform
from .pipeline import TransformPipeline
from .protocol import (
    Transform,
    TransformContext,
    TransformError,
    TransformLookupError,
    TransformRejected,
)
from .registry import TransformRegistry
from .when import when_matches

__all__ = (
    "HttpTransform",
    "NamedTransform",
    "PythonTransform",
    "Transform",
    "TransformContext",
    "TransformError",
    "TransformLookupError",
    "TransformPipeline",
    "TransformRegistry",
    "TransformRejected",
    "when_matches",
)
