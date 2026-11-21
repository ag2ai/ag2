# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""A2A part helpers for A2UI.

Build, identify, and decode A2A ``Part`` objects that carry A2UI JSON.
The A2UI A2A extension (v0.9 and v1.0 alike) says a DataPart's ``data``
"MUST be an array of messages", so a single DataPart carries a JSON array of
A2UI operations. :func:`get_a2ui_data` still decodes a single object, so a
caller can recognise a malformed part; ``create_a2ui_parts(legacy_split=True)``,
which emits one, is deprecated.

The A2UI A2A extension spec marks an A2UI part with
``metadata["mimeType"]``, which is all an A2A v0.3 peer sees: a data part's
``media_type`` does not exist there. Parts are written with both and
recognised by either.
"""

import warnings

from a2a.types import Part

from ag2.a2a.mappers import data_part, is_data_part_with_mime, part_data_to_python, struct_from_dict, struct_to_dict

from .._types import JsonObject, ServerToClientMessage
from ..constants import A2UI_MIME_TYPE

_MIME_TYPE_METADATA_KEY = "mimeType"

LEGACY_SPLIT_DEPRECATION = (
    "create_a2ui_parts(legacy_split=True) is deprecated and will be removed in 1.4.0: the A2UI A2A extension "
    "requires a DataPart's 'data' to be an array of messages."
)


def create_a2ui_parts(
    operations: ServerToClientMessage | list[ServerToClientMessage],
    *,
    legacy_split: bool = False,
) -> list[Part]:
    """Wrap A2UI operations into A2A ``Part`` (s).

    By default produces ONE DataPart whose ``data`` field is the full
    list of operations — the canonical A2A v1.0 encoding for A2UI.

    Args:
        operations: A single A2UI server→client message or a list of them.
        legacy_split: Deprecated, removed in 1.4.0. If True, emit one DataPart
            per operation with ``data`` as a single object, which the A2UI A2A
            extension does not allow in any version.

    Returns:
        A list of ``Part`` objects, each tagged with the A2UI MIME type in
        ``media_type`` and ``metadata["mimeType"]``.
    """
    if isinstance(operations, dict):
        operations = [operations]
    if legacy_split:
        warnings.warn(LEGACY_SPLIT_DEPRECATION, DeprecationWarning, stacklevel=2)
        return [_a2ui_part(op) for op in operations]
    return [_a2ui_part(list(operations))]


def is_a2ui_part(part: Part) -> bool:
    """Check whether an A2A ``Part`` carries A2UI data, by ``media_type`` or ``metadata["mimeType"]``."""
    if is_data_part_with_mime(part, A2UI_MIME_TYPE):
        return True
    return part.HasField("data") and struct_to_dict(part.metadata).get(_MIME_TYPE_METADATA_KEY) == A2UI_MIME_TYPE


def get_a2ui_data(part: Part) -> JsonObject | list[JsonObject] | None:
    """Decode an A2UI DataPart's payload to a native Python value.

    Returns:
        - ``list[dict]`` for canonical A2A v1.0 form (``data = [msg, ...]``).
        - ``dict`` for legacy form (``data = msg``).
        - ``None`` if the part is not A2UI-typed or payload shape is unexpected.
    """
    if not is_a2ui_part(part):
        return None
    data = part_data_to_python(part)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    return None


def _a2ui_part(payload: ServerToClientMessage | list[ServerToClientMessage]) -> Part:
    part = data_part(payload, media_type=A2UI_MIME_TYPE)
    part.metadata.CopyFrom(struct_from_dict({_MIME_TYPE_METADATA_KEY: A2UI_MIME_TYPE}))
    return part
