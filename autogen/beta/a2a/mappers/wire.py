# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

METADATA_PREFIX = "ag2_beta_"
"""Namespace prefix for AG2-beta metadata keys on Parts/Artifacts/Messages."""

PART_KIND_KEY = METADATA_PREFIX + "kind"
"""``Part.metadata`` marker for the original ``Input`` subtype (text/binary/url/...)."""

BINARY_TYPE_KEY = METADATA_PREFIX + "binary_type"
"""``Part.metadata`` marker for ``BinaryInput.kind`` / ``UrlInput.kind``."""

FILENAME_KEY = METADATA_PREFIX + "filename"
"""``Part.metadata`` marker for filename on binary/file-id inputs."""

VENDOR_METADATA_KEY = METADATA_PREFIX + "vendor_metadata"
"""``Part.metadata`` marker for opaque provider blob (e.g. Anthropic file-cache markers)."""

USAGE_METADATA_KEY = METADATA_PREFIX + "usage"
"""``Artifact.metadata`` key carrying the final-chunk ``Usage`` payload.

Schema: ``{"prompt_tokens": int|None, "completion_tokens": int|None,
"total_tokens": int|None, "cache_read_input_tokens": int|None,
"cache_creation_input_tokens": int|None}``.
"""

FINISH_REASON_METADATA_KEY = METADATA_PREFIX + "finish_reason"
"""``Artifact.metadata`` key carrying the OpenAI-style ``finish_reason`` string."""

MODEL_METADATA_KEY = METADATA_PREFIX + "model"
"""``Artifact.metadata`` key carrying the upstream model identifier string."""

REASONING_ARTIFACT_NAME = "reasoning"
"""Dedicated ``Artifact.name`` used to stream ``ModelReasoning`` content
parallel to the main ``result`` artifact."""

RESULT_ARTIFACT_NAME = "result"
"""Default ``Artifact.name`` for the main response body."""

TOOL_CALLS_ARTIFACT_NAME = "tool_calls"
"""Dedicated ``Artifact.name`` carrying server-emitted ``ToolCallEvent`` payloads
when the LLM has chosen to invoke a client-side tool."""

CLIENT_TOOLS_EXTENSION_URI = "https://ag2.dev/a2a/client-tools/v1"
"""URI of the AG2 client-side tools extension. Servers that understand it accept
``Message.metadata[CLIENT_TOOLS_KEY]`` on inbound messages and emit
``ToolCallEvent`` payloads via the ``tool_calls`` artifact + ``input_required``
state when the LLM picks one of the declared client-side tools."""

CLIENT_TOOLS_KEY = METADATA_PREFIX + "client_tools"
"""``Message.metadata`` key carrying the client-declared tool schemas.

Schema: ``[{"name": str, "description": str, "parameters": JSONSchema}, ...]``.
The server registers stub tools under these names before invoking ``Agent.ask``.
"""

TOOL_CALL_REQUEST_KEY = METADATA_PREFIX + "tool_call_request"
"""``Message.metadata`` key on ``input_required`` messages — the server is
asking the client to execute a previously declared client-side tool.

Schema: ``{"id": str, "name": str, "arguments": str (JSON)}``.
"""

TOOL_CALL_RESULT_KEY = METADATA_PREFIX + "tool_call_result"
"""``Message.metadata`` key on follow-up user messages — the client returns the
result of a client-side tool invocation requested via ``TOOL_CALL_REQUEST_KEY``.

Schema: ``{"id": str, "output": str}`` for success or
``{"id": str, "error": str}`` for failure.
"""
