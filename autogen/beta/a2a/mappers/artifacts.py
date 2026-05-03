# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from dataclasses import asdict
from typing import Any

from a2a.types import Artifact, TaskArtifactUpdateEvent

from autogen.beta.events import ModelMessageChunk, Usage

from ._proto import struct_to_dict, value_to_python
from .wire import (
    FINISH_REASON_METADATA_KEY,
    MODEL_METADATA_KEY,
    USAGE_METADATA_KEY,
)


def artifact_text(artifact: Artifact) -> str:
    """Concatenate all text content from an artifact."""
    return "".join(_iter_artifact_text(artifact))


def task_artifact_update_to_events(event: TaskArtifactUpdateEvent) -> Iterator[ModelMessageChunk]:
    """Yield ``ModelMessageChunk`` events for an artifact update.

    Reasoning is no longer carried via artifacts (A2A spec reserves artifacts
    for produced outputs); it now rides the ``status_update`` channel — see
    ``REASONING_KEY`` and ``streams.drain``.
    """
    for text in _iter_artifact_text(event.artifact):
        if not text:
            continue
        yield ModelMessageChunk(text)


def usage_to_metadata(usage: Usage) -> dict[str, Any]:
    """Encode ``Usage`` into a JSON-friendly dict for ``Artifact.metadata``."""
    return {k: v for k, v in asdict(usage).items() if v is not None}


def usage_from_metadata(metadata: dict[str, Any] | None) -> Usage:
    """Read ``Usage`` from already-decoded ``Artifact.metadata`` (returns empty ``Usage`` if missing).

    Numeric token counts come back from proto ``Struct`` as ``float`` (see
    ``_proto`` module warning); each is cast to ``int`` to preserve the
    ``Usage`` contract.
    """
    if not metadata:
        return Usage()
    raw = metadata.get(USAGE_METADATA_KEY) or {}
    if not isinstance(raw, dict):
        return Usage()
    return Usage(
        prompt_tokens=_to_int(raw.get("prompt_tokens")),
        completion_tokens=_to_int(raw.get("completion_tokens")),
        total_tokens=_to_int(raw.get("total_tokens")),
        cache_read_input_tokens=_to_int(raw.get("cache_read_input_tokens")),
        cache_creation_input_tokens=_to_int(raw.get("cache_creation_input_tokens")),
    )


def _to_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def finish_reason_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Read OpenAI-style ``finish_reason`` from already-decoded ``Artifact.metadata``."""
    if not metadata:
        return None
    value = metadata.get(FINISH_REASON_METADATA_KEY)
    return value if isinstance(value, str) else None


def model_from_metadata(metadata: dict[str, Any] | None) -> str | None:
    """Read upstream model identifier from already-decoded ``Artifact.metadata``."""
    if not metadata:
        return None
    value = metadata.get(MODEL_METADATA_KEY)
    return value if isinstance(value, str) else None


def build_result_metadata(
    *,
    usage: Usage | None,
    finish_reason: str | None,
    model: str | None,
) -> dict[str, Any] | None:
    """Compose ``Artifact.metadata`` payload for the final ``result`` chunk."""
    payload: dict[str, Any] = {}
    if usage is not None and usage:
        payload[USAGE_METADATA_KEY] = usage_to_metadata(usage)
    if finish_reason:
        payload[FINISH_REASON_METADATA_KEY] = finish_reason
    if model:
        payload[MODEL_METADATA_KEY] = model
    return payload or None


def artifact_metadata_dict(artifact: Artifact) -> dict[str, Any]:
    """Decode the ``Struct`` metadata of an Artifact to a plain dict."""
    return struct_to_dict(artifact.metadata) if artifact.HasField("metadata") else {}


def _iter_artifact_text(artifact: Artifact) -> Iterator[str]:
    for part in artifact.parts:
        oneof = part.WhichOneof("content")
        if oneof == "text":
            yield part.text
        elif oneof == "data":
            # Cross-compat: legacy `autogen/a2a/` emits streaming chunks as
            # ``DataPart(data={"content": text})`` — accept that shape too.
            decoded = value_to_python(part.data)
            if isinstance(decoded, dict):
                content = decoded.get("content")
                if isinstance(content, str):
                    yield content
