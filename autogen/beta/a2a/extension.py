# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# AG2 client-tools extension: announced via the URI on the AgentCard,
# wire-tagged via the MIME types below. Private to AG2 — never sent to
# non-AG2 servers, never inspected by intermediaries.
EXTENSION_URI = "urn:ag2:client-tools:v1"

MIME_TOOL_SCHEMAS = "application/vnd.ag2.tool-schemas+json"
MIME_TOOL_CALL = "application/vnd.ag2.tool-call+json"
MIME_TOOL_RESULT = "application/vnd.ag2.tool-result+json"
MIME_HISTORY = "application/vnd.ag2.history+json"

# Bidirectional context-variables sync rides on Message.metadata under this key.
CONTEXT_UPDATE_METADATA_KEY = "ag2.context_update"

# Dependency key for splicing extra A2A ``Part``s onto the outgoing message.
EXTRA_PARTS_DEPENDENCY_KEY = "a2a:extra_parts"

# Per-call tenant override in ``context.variables`` — wins over ``A2AConfig.tenant``.
TENANT_VARIABLE_KEY = "a2a:tenant"
