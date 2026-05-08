# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

EXTENSION_URI = "urn:ag2:client-tools:v1"

MIME_TOOL_SCHEMAS = "application/vnd.ag2.tool-schemas+json"
MIME_TOOL_CALL = "application/vnd.ag2.tool-call+json"
MIME_TOOL_RESULT = "application/vnd.ag2.tool-result+json"
MIME_HISTORY = "application/vnd.ag2.history+json"

# Bidirectional context-variables sync rides on Message.metadata under this key.
# Server -> client on the final agent message; client -> server on user messages.
CONTEXT_UPDATE_METADATA_KEY = "ag2.context_update"

# Dependency key the client reads to splice extra A2A ``Part``s onto the
# outgoing message. Lets users smuggle protocol-level Parts (e.g. for
# experimental extensions) without changing public ``A2AConfig`` shape.
EXTRA_PARTS_DEPENDENCY_KEY = "a2a:extra_parts"

# Per-call tenant override. When set in ``context.variables`` it wins over the
# ``A2AConfig.tenant`` default — lets a single Agent instance fan out to
# multiple tenants without rebuilding its config.
TENANT_VARIABLE_KEY = "a2a:tenant"
