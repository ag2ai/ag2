# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

CONTEXT_ID_VAR_KEY = "ag:a2a:context_id"
"""Key under which the server-issued A2A context_id is stored in `Context.variables`."""

TASK_ID_VAR_KEY = "ag:a2a:task_id"
"""Key under which the current A2A task_id is stored in `Context.variables`."""

RESULT_ARTIFACT_NAME = "result"
"""Name of the artifact that carries the agent's final reply."""

AG2_BETA_METADATA_KEY_PREFIX = "ag2_beta_"
"""Prefix used in `Part.metadata` to namespace ag2-beta-specific markers and avoid
collisions with the legacy `ag2_` prefix used by `autogen.a2a`."""

PART_KIND_METADATA_KEY = AG2_BETA_METADATA_KEY_PREFIX + "kind"
"""`Part.metadata` key that records the original Input subtype, so we can roundtrip
back into the right Input class on the receiving side (e.g. `file_id`, `url`, `binary`).
"""

PROVIDER_NAME = "a2a"
"""Value used in `ModelResponse.provider` for replies received over A2A."""
