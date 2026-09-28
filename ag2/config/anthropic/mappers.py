# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import logging
from collections.abc import Iterable, Sequence
from typing import Any, Literal, TypeAlias, TypeGuard, cast, get_args

from anthropic.types import (
    CacheControlEphemeralParam,
    CitationsConfigParam,
    CodeExecutionTool20260521Param,
    ContentBlockParam,
    DirectCallerParam,
    DocumentBlockParam,
    ImageBlockParam,
    MemoryTool20250818Param,
    MessageParam,
    ServerToolCaller20260120Param,
    ServerToolCallerParam,
    TextBlockParam,
    ToolBash20250124Param,
    ToolParam,
    ToolResultBlockParam,
    ToolSearchToolBm25_20251119Param,
    ToolSearchToolRegex20251119Param,
    ToolUseBlockParam,
    UserLocationParam,
    WebFetchTool20260318Param,
    WebFetchURLSourceAllParam,
    WebFetchURLSourceExceptParam,
    WebFetchURLSourceNoneParam,
    WebFetchURLSourceOnlyParam,
    WebFetchURLSourceToolReferenceParam,
    WebFetchURLSourcesParam,
    WebSearchTool20260318Param,
)
from anthropic.types.tool_result_block_param import Content as ToolResultContent
from anthropic.types.tool_use_block_param import Caller
from fast_depends.library.serializer import SerializerProto

from ag2.compact import CompactionSummary
from ag2.config.anthropic.events import (
    AnthropicRedactedThinkingEvent,
    AnthropicServerToolCallEvent,
    AnthropicServerToolResultEvent,
)
from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    DataInput,
    FileIdInput,
    ModelRequest,
    ModelResponse,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    ToolResultsEvent,
    UrlInput,
    Usage,
)

logger = logging.getLogger(__name__)

from ag2.exceptions import (
    UnsupportedInputError,
    UnsupportedToolError,
    WebFetchOptionUnsupportedError,
    WebFetchUrlSourceToolNotFoundError,
)
from ag2.files import FileProvider
from ag2.response import ResponseProto
from ag2.tools.builtin.anthropic_bash import ANTHROPIC_BASH_TOOL_NAME, AnthropicBashToolSchema
from ag2.tools.builtin.code_execution import CodeExecutionToolSchema
from ag2.tools.builtin.mcp_server import MCPServerToolSchema
from ag2.tools.builtin.memory import MemoryToolSchema
from ag2.tools.builtin.shell import ShellToolSchema
from ag2.tools.builtin.skills import SkillsToolSchema
from ag2.tools.builtin.tool_search import ToolSearchToolSchema
from ag2.tools.builtin.web_fetch import (
    WEB_FETCH_TOOL_NAME,
    ExceptTools,
    OnlyTools,
    ToolResultSources,
    UrlSources,
    WebFetchToolSchema,
    WebFetchVersions,
)
from ag2.tools.builtin.web_search import UserLocation, WebSearchToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.schemas import ToolSchema

#: What a source with no tools to name may contribute: everything or nothing.
UnfilteredUrlSourceParam: TypeAlias = WebFetchURLSourceAllParam | WebFetchURLSourceNoneParam

#: One entry of ``url_sources``: what a single source may contribute, as the API discriminates it.
UrlSourceParam: TypeAlias = UnfilteredUrlSourceParam | WebFetchURLSourceOnlyParam | WebFetchURLSourceExceptParam

#: The two tool search variants, each carrying its own ``name``/``type`` pairing.
ToolSearchParam: TypeAlias = ToolSearchToolRegex20251119Param | ToolSearchToolBm25_20251119Param

WEB_FETCH_USE_CACHE_SINCE: WebFetchVersions = "web_fetch_20260309"
WEB_FETCH_RESPONSE_INCLUSION_SINCE: WebFetchVersions = "web_fetch_20260318"


def _ensure_additional_properties_false(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively add additionalProperties: false to all object schemas.

    Anthropic requires this on every object node in output_config.format.schema.
    """
    schema = dict(schema)

    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)

    if "properties" in schema:
        schema["properties"] = {
            k: _ensure_additional_properties_false(v) if isinstance(v, dict) else v
            for k, v in schema["properties"].items()
        }

    if "$defs" in schema:
        schema["$defs"] = {
            k: _ensure_additional_properties_false(v) if isinstance(v, dict) else v for k, v in schema["$defs"].items()
        }

    for key in ("anyOf", "oneOf", "allOf"):
        if key in schema:
            schema[key] = [
                _ensure_additional_properties_false(item) if isinstance(item, dict) else item for item in schema[key]
            ]

    if "items" in schema and isinstance(schema["items"], dict):
        schema["items"] = _ensure_additional_properties_false(schema["items"])

    return schema


def response_proto_to_output_config(response: ResponseProto | None) -> dict[str, Any] | None:
    """Convert a ResponseProto to Anthropic output_config."""
    if not response or not response.json_schema:
        return None

    strict_schema = _ensure_additional_properties_false(response.json_schema)

    return {
        "format": {
            "type": "json_schema",
            "schema": strict_schema,
        },
    }


def _ensure_object_schema(params: dict[str, Any]) -> dict[str, Any]:
    """Anthropic requires input_schema to be type: object."""
    schema = dict(params)
    schema["type"] = "object"
    schema.setdefault("properties", {})
    return schema


def _reject_unsupported_web_fetch_options(t: WebFetchToolSchema) -> None:
    """Refuse a web fetch option the selected tool version predates.

    Measured 2026-09-19: the API rejects the same pairing with
    ``Extra inputs are not permitted``, naming no version that would take it. Refusing here
    spends no request and says which version the option arrived in.
    """
    if t.use_cache is not None and t.web_fetch_version < WEB_FETCH_USE_CACHE_SINCE:
        raise WebFetchOptionUnsupportedError("use_cache", t.web_fetch_version, WEB_FETCH_USE_CACHE_SINCE)

    if t.response_inclusion is not None and t.web_fetch_version < WEB_FETCH_RESPONSE_INCLUSION_SINCE:
        raise WebFetchOptionUnsupportedError(
            "response_inclusion", t.web_fetch_version, WEB_FETCH_RESPONSE_INCLUSION_SINCE
        )


def _tool_references(names: Iterable[str]) -> list[WebFetchURLSourceToolReferenceParam]:
    """Name each tool the way a `url_sources` filter refers to it."""
    return [{"type": "tool_reference", "name": n} for n in names]


def _unfiltered_source_to_api(source: Literal["all", "none"]) -> UnfilteredUrlSourceParam:
    """Tag a source that names no tools: everything it supplies, or nothing.

    ``UrlSources`` validates nothing, so a value outside the pair is sent as written: this is a
    policy control, and a 400 naming the typo beats quietly substituting the closed policy.
    """
    if source == "all":
        return {"type": "all"}

    if source == "none":
        return {"type": "none"}

    return cast(UnfilteredUrlSourceParam, {"type": source})


def _url_source_to_api(source: ToolResultSources) -> UrlSourceParam:
    """Tag one `url_sources` entry the way the API discriminates it."""
    if isinstance(source, OnlyTools):
        return {"type": "only", "tools": _tool_references(source.tools)}

    if isinstance(source, ExceptTools):
        return {"type": "except", "tools": _tool_references(source.tools)}

    return _unfiltered_source_to_api(source)


def _url_sources_to_api(url_sources: UrlSources) -> WebFetchURLSourcesParam:
    """Map the sources that were set. One left unset is the API's default, not a decision to send."""
    result: WebFetchURLSourcesParam = {}
    if url_sources.user_input is not None:
        result["user_input"] = _unfiltered_source_to_api(url_sources.user_input)
    if url_sources.client_tool_results is not None:
        result["client_tool_results"] = _url_source_to_api(url_sources.client_tool_results)
    if url_sources.server_tool_results is not None:
        result["server_tool_results"] = _url_source_to_api(url_sources.server_tool_results)
    return result


def _web_fetch_tool_to_api(t: WebFetchToolSchema) -> WebFetchTool20260318Param:
    """Build the web fetch entry against the newest version's param, whichever version is sent.

    Its fields are a superset of every older one's, and the gate above has already refused any
    option the selected version predates.
    """
    result: WebFetchTool20260318Param = {
        # The four params differ only in this literal, so the newest stands in as their
        # superset and the version is carried through unchecked.
        "type": cast(Literal["web_fetch_20260318"], t.web_fetch_version),
        "name": WEB_FETCH_TOOL_NAME,
    }
    if t.max_uses is not None:
        result["max_uses"] = t.max_uses
    if t.allowed_domains is not None:
        result["allowed_domains"] = t.allowed_domains
    if t.blocked_domains is not None:
        result["blocked_domains"] = t.blocked_domains
    if t.citations is not None:
        result["citations"] = {"enabled": t.citations}
    if t.max_content_tokens is not None:
        result["max_content_tokens"] = t.max_content_tokens
    if t.strict is not None:
        result["strict"] = t.strict
    if t.use_cache is not None:
        result["use_cache"] = t.use_cache
    if t.response_inclusion is not None:
        result["response_inclusion"] = t.response_inclusion
    if t.url_sources is not None and (sources := _url_sources_to_api(t.url_sources)):
        result["url_sources"] = sources
    return result


def _user_location_to_api(location: UserLocation) -> UserLocationParam:
    """Tag the location the way the API discriminates it; only the fields that were set travel."""
    result: UserLocationParam = {"type": "approximate"}
    if location.city is not None:
        result["city"] = location.city
    if location.region is not None:
        result["region"] = location.region
    if location.country is not None:
        result["country"] = location.country
    if location.timezone is not None:
        result["timezone"] = location.timezone
    return result


def _web_search_tool_to_api(t: WebSearchToolSchema) -> WebSearchTool20260318Param:
    """Build the web search entry against the newest version's param, whichever version is sent."""
    result: WebSearchTool20260318Param = {
        # The three params differ only in this literal, so the newest stands in as their
        # superset and the version is carried through unchecked.
        "type": cast(Literal["web_search_20260318"], t.web_search_version),
        "name": "web_search",
    }
    if t.max_uses is not None:
        result["max_uses"] = t.max_uses
    if t.user_location is not None:
        result["user_location"] = _user_location_to_api(t.user_location)
    if t.allowed_domains is not None:
        result["allowed_domains"] = t.allowed_domains
    if t.blocked_domains is not None:
        result["blocked_domains"] = t.blocked_domains
    return result


def _tool_search_tool_to_api(t: ToolSearchToolSchema) -> ToolSearchParam:
    """Build the tool search entry from the variant's own param.

    The ``name`` is the variant id — the API rejects a generic "tool_search", and this is the
    name it puts on the ``server_tool_use`` block it sends back. The SDK's ``type`` also accepts
    the undated form (``tool_search_tool_regex``); the dated one is what ag2 has always sent.
    """
    if t.mode == "bm25":
        bm25: ToolSearchToolBm25_20251119Param = {
            "type": "tool_search_tool_bm25_20251119",
            "name": "tool_search_tool_bm25",
        }
        return bm25

    regex: ToolSearchToolRegex20251119Param = {
        "type": "tool_search_tool_regex_20251119",
        "name": "tool_search_tool_regex",
    }
    return regex


def tool_to_api(t: ToolSchema) -> dict[str, Any]:
    if isinstance(t, FunctionToolSchema):
        fn_tool: ToolParam = {
            "name": t.function.name,
            "description": t.function.description,
            "input_schema": _ensure_object_schema(t.function.parameters),
        }
        if t.defer_loading:
            fn_tool["defer_loading"] = True
        return dict(fn_tool)

    elif isinstance(t, WebSearchToolSchema):
        return dict(_web_search_tool_to_api(t))

    elif isinstance(t, CodeExecutionToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
        # The versioned params differ only in the `type` literal, so the newest stands in
        # for all of them and the version is carried through unchecked.
        code_execution: CodeExecutionTool20260521Param = {
            "type": cast(Literal["code_execution_20260521"], t.version),
            "name": "code_execution",
        }
        return dict(code_execution)

    elif isinstance(t, WebFetchToolSchema):
        _reject_unsupported_web_fetch_options(t)
        # A TypedDict is a plain dict at runtime; the copy only widens the static type.
        return dict(_web_fetch_tool_to_api(t))

    elif isinstance(t, MemoryToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
        memory: MemoryTool20250818Param = {"type": t.version, "name": "memory"}
        return dict(memory)

    elif isinstance(t, AnthropicBashToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
        # Client-executed: Anthropic returns a plain tool_use block and waits for
        # a tool_result. AnthropicBashTool registers the executor under the same
        # name, so the call is answered instead of raising ToolNotFound.
        bash: ToolBash20250124Param = {"type": t.version, "name": ANTHROPIC_BASH_TOOL_NAME}
        return dict(bash)

    elif isinstance(t, ShellToolSchema):
        # Anthropic's bash tool is client-side — it ships a typed schema but the
        # application must execute the command itself and return a tool_result.
        # ag2 does not provide a default executor for this here.
        # Use SandboxShellTool (tools/shell/) instead, which runs commands via subprocess
        # and works with any provider.
        raise UnsupportedToolError(t.type, "anthropic")

    elif isinstance(t, SkillsToolSchema):
        # Skills are handled via the container parameter, not the tools[] array.
        # Use extract_skills_for_container() in the client instead.
        raise UnsupportedToolError(t.type, "anthropic")

    elif isinstance(t, MCPServerToolSchema):
        # https://platform.claude.com/docs/en/docs/agents-and-tools/mcp-connector
        # The one entry built by hand: `anthropic` ships no param type for an
        # mcp_toolset, so there is nothing to check this shape against.
        result: dict[str, Any] = {
            "type": "mcp_toolset",
            "mcp_server_name": t.server_label,
        }
        if t.allowed_tools is not None:
            result["default_config"] = {"enabled": False}
            result["configs"] = {name: {"enabled": True} for name in t.allowed_tools}
        if t.blocked_tools is not None:
            configs: dict[str, Any] = result.get("configs", {})
            configs.update({name: {"enabled": False} for name in t.blocked_tools})
            result["configs"] = configs
        return result

    elif isinstance(t, ToolSearchToolSchema):
        # https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool
        return dict(_tool_search_tool_to_api(t))

    raise UnsupportedToolError(t.type, "anthropic")


def _url_source_tool_names(source: ToolResultSources | None) -> tuple[str, ...]:
    """The tool names a filter resolves against ``tools[]``. ``all`` and ``none`` name nothing."""
    if isinstance(source, (OnlyTools, ExceptTools)):
        return tuple(source.tools)

    return ()


def _declared_tool_names(mapped: Sequence[dict[str, Any]]) -> set[str]:
    """Every tool name this request puts on the wire, for a ``url_sources`` filter to resolve against.

    An MCP toolset carries no ``name`` of its own; the tools it enables are the ``configs`` entries
    it switches on. They are named in the body, so a filter may name them too.
    """
    names: set[str] = set()

    for entry in mapped:
        if isinstance(name := entry.get("name"), str):
            names.add(name)

        configs = entry.get("configs")
        if isinstance(configs, dict):
            names.update(n for n, config in configs.items() if config.get("enabled"))

    return names


def _reject_unknown_url_source_tools(tools: Sequence[ToolSchema], mapped: Sequence[dict[str, Any]]) -> None:
    """Refuse a ``url_sources`` filter naming a tool this request does not declare.

    Checked against the mapped entries rather than the schemas, so what counts as declared is
    exactly what goes on the wire. Declaration is all that is checked: a name is not matched
    against the side of ``url_sources`` it was written under, because whether a tool contributes
    as a client or a server result is the provider's call, not one ag2 can make for it.
    """
    declared = _declared_tool_names(mapped)

    for t in tools:
        if not isinstance(t, WebFetchToolSchema) or t.url_sources is None:
            continue

        for source in (t.url_sources.client_tool_results, t.url_sources.server_tool_results):
            for tool_name in _url_source_tool_names(source):
                if tool_name not in declared:
                    raise WebFetchUrlSourceToolNotFoundError(tool_name, sorted(declared))


def tools_to_api(tools: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    """Map a whole request's tools, enforcing what only the whole request can be checked against."""
    mapped = [tool_to_api(t) for t in tools]
    _reject_unknown_url_source_tools(tools, mapped)
    return mapped


def extract_mcp_servers(tools: Iterable[ToolSchema]) -> list[dict[str, Any]]:
    """Extract Anthropic mcp_servers definitions from MCPServerToolSchema instances."""
    servers: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, MCPServerToolSchema):
            server: dict[str, Any] = {
                "type": "url",
                "url": t.server_url,
                "name": t.server_label,
            }
            if t.authorization_token is not None:
                server["authorization_token"] = t.authorization_token
            servers.append(server)
    return servers


def extract_skills_for_container(tools: Iterable[ToolSchema]) -> list[dict[str, Any]]:
    """Extract Anthropic skills from SkillsToolSchema instances for the container parameter."""
    skills: list[dict[str, Any]] = []
    for t in tools:
        if isinstance(t, SkillsToolSchema):
            for s in t.skills:
                entry: dict[str, Any] = {
                    "type": "anthropic",
                    "skill_id": s.id,
                    "version": str(s.version) if s.version is not None else "latest",
                }
                skills.append(entry)
    return skills


# `anthropic` 1.x dropped these from the Messages API signature; the API still reads them from the request body.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k")


def take_sampling_fields(values: dict[str, Any]) -> dict[str, Any]:
    """Pop every sampling key from ``values``, an explicit ``None`` included, returning the ones that were set."""
    return {name: value for name in SAMPLING_FIELDS if (value := values.pop(name, None)) is not None}


def merge_sampling_into_extra_body(
    sampling: dict[str, Any], extra_body: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Fold the sampling values under an explicit ``extra_body``, which wins on collision — as with ``mcp_servers``."""
    if not sampling:
        return extra_body
    return {**sampling, **(extra_body or {})}


_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp"})

_ImageMediaType: TypeAlias = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
"""The only image media types the API accepts inline; anything else is answered with a 400."""

_CacheTtl: TypeAlias = Literal["5m", "1h"]

_Role: TypeAlias = Literal["user", "assistant"]


def _is_image_media_type(media_type: str) -> TypeGuard[_ImageMediaType]:
    return media_type in get_args(_ImageMediaType)


def _is_cache_ttl(value: str) -> TypeGuard[_CacheTtl]:
    return value in get_args(_CacheTtl)


def _file_id_block(file_id: str, filename: str | None) -> ImageBlockParam | DocumentBlockParam:
    """Reference an uploaded file; the block kind is inferred from the filename extension."""
    if filename and "." in filename and f".{filename.rsplit('.', 1)[-1].lower()}" in _IMAGE_EXTENSIONS:
        image: ImageBlockParam = {"type": "image", "source": {"type": "file", "file_id": file_id}}
        return image
    document: DocumentBlockParam = {"type": "document", "source": {"type": "file", "file_id": file_id}}
    return document


def _url_block(inp: UrlInput) -> ImageBlockParam | DocumentBlockParam:
    if inp.kind is BinaryType.IMAGE:
        image: ImageBlockParam = {"type": "image", "source": {"type": "url", "url": inp.url}}
        return image
    if inp.kind in (BinaryType.DOCUMENT, BinaryType.BINARY):
        document: DocumentBlockParam = {"type": "document", "source": {"type": "url", "url": inp.url}}
        return document
    raise UnsupportedInputError(f"UrlInput({inp.kind.value})", "anthropic")


def _image_block(inp: BinaryInput) -> ImageBlockParam:
    media_type = inp.media_type
    if not _is_image_media_type(media_type):
        raise UnsupportedInputError(f"BinaryInput(image, media_type={media_type})", "anthropic")
    b64 = base64.b64encode(inp.data).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}}


def _document_block(inp: BinaryInput) -> DocumentBlockParam:
    """Inline a document: a PDF as base64, plain text as text. The API refuses base64 for anything but a PDF."""
    if inp.media_type == "application/pdf":
        b64 = base64.b64encode(inp.data).decode()
        return {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": b64}}
    if inp.media_type == "text/plain":
        try:
            text = inp.data.decode("utf-8")
        except UnicodeDecodeError:
            raise UnsupportedInputError(
                "BinaryInput(document, media_type=text/plain, not UTF-8)", "anthropic"
            ) from None
        return {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": text}}
    raise UnsupportedInputError(f"BinaryInput(document, media_type={inp.media_type})", "anthropic")


def _cache_control(inp: BinaryInput) -> CacheControlEphemeralParam | None:
    """The ``cache_control`` a caller set in ``vendor_metadata``, in the shape the API accepts."""
    value = inp.vendor_metadata.get("cache_control")
    if value is None:
        return None
    if isinstance(value, dict) and value.get("type") == "ephemeral" and set(value) <= {"type", "ttl"}:
        cache_control: CacheControlEphemeralParam = {"type": "ephemeral"}
        if "ttl" not in value:
            return cache_control
        if isinstance(ttl := value["ttl"], str) and _is_cache_ttl(ttl):
            cache_control["ttl"] = ttl
            return cache_control
    raise UnsupportedInputError(f"BinaryInput(cache_control={value!r})", "anthropic")


def _citations(inp: BinaryInput) -> CitationsConfigParam | None:
    """The ``citations`` a caller set in ``vendor_metadata``, in the shape the API accepts."""
    value = inp.vendor_metadata.get("citations")
    if value is None:
        return None
    if isinstance(value, dict) and set(value) <= {"enabled"}:
        citations: CitationsConfigParam = {}
        if "enabled" not in value:
            return citations
        if isinstance(enabled := value["enabled"], bool):
            citations["enabled"] = enabled
            return citations
    raise UnsupportedInputError(f"BinaryInput(citations={value!r})", "anthropic")


def _caller(call: ToolCallEvent) -> Caller | None:
    """The ``caller`` recorded on a tool call, rebuilt as one of the kinds the SDK models."""
    value = call.vendor_metadata.get("caller")
    if value is None:
        return None
    if isinstance(value, dict):
        kind = value.get("type")
        if kind == "direct" and set(value) == {"type"}:
            direct: DirectCallerParam = {"type": "direct"}
            return direct
        if set(value) == {"type", "tool_id"} and isinstance(tool_id := value["tool_id"], str):
            if kind == "code_execution_20250825":
                caller: ServerToolCallerParam = {"type": "code_execution_20250825", "tool_id": tool_id}
                return caller
            if kind == "code_execution_20260120":
                caller_20260120: ServerToolCaller20260120Param = {"type": "code_execution_20260120", "tool_id": tool_id}
                return caller_20260120
    raise UnsupportedInputError(f"ToolCallEvent(caller={value!r})", "anthropic")


def _tool_result_block(
    result: "ToolResultEvent",
    serializer: SerializerProto,
    toolset_name: str | None = None,
) -> ToolResultBlockParam:
    """Render one tool result as an Anthropic ``tool_result`` block.

    Shared by the wrapped (``ToolResultsEvent``) and loose paths so a result
    renders identically whichever one carries it.
    """
    parts: list[ToolResultContent] = []
    for part in result.result.parts:
        if isinstance(part, TextInput):
            parts.append(_text_block(part.content))
        elif isinstance(part, DataInput):
            parts.append(_text_block(serializer.encode(part.data).decode()))
        elif isinstance(part, BinaryInput):
            if part.kind is BinaryType.IMAGE:
                parts.append(_image_block(part))
            elif part.kind is BinaryType.DOCUMENT:
                parts.append(_document_block(part))
            else:
                raise UnsupportedInputError(f"BinaryInput({part.kind.value})", "anthropic")
        elif isinstance(part, UrlInput):
            parts.append(_url_block(part))
        elif isinstance(part, FileIdInput):
            parts.append(_file_id_block(part.file_id, part.filename))
        else:
            raise UnsupportedInputError(type(part).__name__, "anthropic")

    block: ToolResultBlockParam = {
        "type": "tool_result",
        "tool_use_id": result.parent_id,
        "content": only["text"] if len(parts) == 1 and (only := parts[0])["type"] == "text" else parts,
    }
    if isinstance(result, ToolErrorEvent):
        # Without this the model reads a traceback as a successful result.
        block["is_error"] = True
    if toolset_name is not None:
        # The API refuses a tool_result answering a member tool_use that does not
        # repeat the paired tool_use's toolset_name.
        block["toolset_name"] = toolset_name
    return block


def _text_block(text: str) -> TextBlockParam:
    return {"type": "text", "text": text}


def has_file_id_references(messages: Iterable[BaseEvent]) -> bool:
    """True if any message (user turn or tool result) references a file_id.

    Used by the client to auto-inject the `files-api-2025-04-14` beta header.
    """
    for msg in messages:
        if isinstance(msg, ModelRequest):
            if any(isinstance(p, FileIdInput) for p in msg.parts):
                return True
        elif isinstance(msg, ToolResultsEvent):
            for r in msg.results:
                if any(isinstance(p, FileIdInput) for p in r.result.parts):
                    return True
    return False


def convert_messages(
    messages: Iterable[BaseEvent],
    serializer: SerializerProto,
) -> list[MessageParam]:
    event_list = list(messages)

    # Collect all tool_use IDs present in the conversation so we can
    # drop orphaned tool_result blocks whose matching tool_use was
    # trimmed by a reduction policy (SlidingWindow, TokenBudget, etc.).
    valid_tool_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ModelResponse):
            for call in message.tool_calls.calls:
                valid_tool_ids.add(call.id)

    # Collect all parent_ids referenced by ToolResultsEvent blocks so we
    # can also drop orphaned tool_use blocks — the mirror case of the
    # above. An orphan tool_use with no matching tool_result makes the
    # payload invalid under Anthropic's API contract ("`tool_use` ids
    # were found without `tool_result` blocks immediately after"). This
    # happens when:
    #   - A ToolResultsEvent failed to persist (crash mid-turn, storage
    #     failure, concurrent write on a shared stream).
    #   - Compaction/reduction kept the ModelResponse(tool_use) but
    #     dropped the following ToolResultsEvent.
    # Rather than fail the whole conversation, skip unresolved tool_use
    # blocks. Any accompanying assistant text is still delivered.
    resolved_tool_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ToolResultsEvent):
            for r in message.results:
                parent = getattr(r, "parent_id", None)
                if parent:
                    resolved_tool_ids.add(parent)
        # Loose ToolResultEvent / ToolErrorEvent entries appear when the
        # ToolResultsEvent wrapper failed to save. Treat them as
        # resolving their parent so the tool_use stays valid.
        elif isinstance(message, (ToolResultEvent, ToolErrorEvent)):
            parent = getattr(message, "parent_id", None)
            if parent:
                resolved_tool_ids.add(parent)

    # A member tool_use's toolset_name has to be repeated on its tool_result. Read only for
    # the tool_use blocks that are emitted, so an orphan is still dropped rather than refused.
    toolset_names: dict[str, str] = {}
    for message in event_list:
        if isinstance(message, ModelResponse):
            for call in message.tool_calls.calls:
                if call.id not in resolved_tool_ids or (name := call.vendor_metadata.get("toolset_name")) is None:
                    continue
                if not isinstance(name, str):
                    raise UnsupportedInputError(f"ToolCallEvent(toolset_name={name!r})", "anthropic")
                toolset_names[call.id] = name

    # Content is gathered per turn and each message is built once at the end: the SDK
    # declares `content` as an `Iterable`, so a built message cannot be appended to.
    turns: list[tuple[_Role, str | list[ContentBlockParam]]] = []
    # Track tool_use_ids we've emitted tool_result blocks for, so the
    # individual-ToolResultEvent fallback below doesn't double-emit when
    # both the wrapper and the leaves are present. Pre-populate from any
    # ToolResultsEvent wrappers we'll encounter — individuals arrive in
    # event_list BEFORE the wrapper that aggregates them, so without this
    # pre-scan the fallback branch would emit first and the wrapper would
    # emit again, yielding duplicate tool_result blocks for the same
    # tool_use_id.
    emitted_result_ids: set[str] = set()
    for message in event_list:
        if isinstance(message, ToolResultsEvent):
            for r in message.results:
                if r.parent_id in valid_tool_ids:
                    emitted_result_ids.add(r.parent_id)

    for message in event_list:
        if isinstance(message, ModelResponse):
            content: list[ContentBlockParam] = []
            if message.message:
                content.append(_text_block(message.message.content))
            # Skip tool_use blocks whose matching tool_result is missing
            # from the event list. See the `resolved_tool_ids` block above
            # for why this asymmetry exists. Keeping the assistant's text
            # (if any) means the model's reasoning is preserved even when
            # the tool execution record is lost.
            for call in message.tool_calls.calls:
                if call.id not in resolved_tool_ids:
                    logger.warning(
                        "Dropping orphan tool_use id=%s name=%s (no matching tool_result). "
                        "See mappers.py comment for context.",
                        call.id,
                        call.name,
                    )
                    continue
                use_block: ToolUseBlockParam = {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": json.loads(call.arguments or "{}"),
                }
                if (caller := _caller(call)) is not None:
                    use_block["caller"] = caller
                if (toolset_name := toolset_names.get(call.id)) is not None:
                    use_block["toolset_name"] = toolset_name
                content.append(use_block)
            if content:
                turns.append(("assistant", content))

        # AnthropicRedactedThinkingEvent rides along: Anthropic requires the block
        # echoed back unchanged. AnthropicContainerUploadEvent deliberately has no
        # branch — the API refuses container_upload inside an assistant turn.
        elif isinstance(
            message,
            (AnthropicServerToolCallEvent, AnthropicServerToolResultEvent, AnthropicRedactedThinkingEvent),
        ):
            # The one untyped boundary: the dump of the SDK's own response block, which
            # pydantic validated on the way in, is the param block of the same kind.
            block = cast(ContentBlockParam, message.block.model_dump(exclude_none=True, mode="json"))
            if turns and turns[-1][0] == "assistant" and isinstance(open_turn := turns[-1][1], list):
                open_turn.append(block)
            else:
                turns.append(("assistant", [block]))

        elif isinstance(message, ToolResultsEvent):
            tool_results: list[ContentBlockParam] = []
            for r in message.results:
                # Drop orphan tool_result whose matching tool_use was
                # trimmed by a reduction policy (SlidingWindow, etc.).
                # If the conversation has no tool_use blocks at all,
                # skip the filter — caller passed only tool_results
                # (e.g. unit-testing the rendering in isolation).
                if valid_tool_ids and r.parent_id not in valid_tool_ids:
                    continue
                tool_results.append(_tool_result_block(r, serializer, toolset_names.get(r.parent_id)))
                emitted_result_ids.add(r.parent_id)
            if tool_results:
                turns.append(("user", tool_results))

        elif isinstance(message, ModelRequest):
            content_parts: list[ContentBlockParam] = []
            for inp in message.parts:
                if isinstance(inp, TextInput):
                    content_parts.append(_text_block(inp.content))

                elif isinstance(inp, DataInput):
                    content_parts.append(_text_block(serializer.encode(inp.data).decode()))

                elif isinstance(inp, FileIdInput):
                    if (provider := getattr(inp, "provider", None)) and provider is not FileProvider.ANTHROPIC:
                        raise UnsupportedInputError(
                            f"file uploaded via '{provider.value}' cannot be used with '{FileProvider.ANTHROPIC.value}'",
                            "anthropic",
                        )
                    content_parts.append(_file_id_block(inp.file_id, inp.filename))

                elif isinstance(inp, UrlInput):
                    content_parts.append(_url_block(inp))

                elif isinstance(inp, BinaryInput):
                    if inp.kind is BinaryType.IMAGE:
                        image = _image_block(inp)
                        if inp.vendor_metadata.get("citations") is not None:
                            # The API answers `image.citations: Extra inputs are not permitted`.
                            raise UnsupportedInputError("BinaryInput(image, citations)", "anthropic")
                        if (cache_control := _cache_control(inp)) is not None:
                            image["cache_control"] = cache_control
                        content_parts.append(image)

                    elif inp.kind is BinaryType.DOCUMENT:
                        document = _document_block(inp)
                        if (cache_control := _cache_control(inp)) is not None:
                            document["cache_control"] = cache_control
                        if (citations := _citations(inp)) is not None:
                            document["citations"] = citations
                        content_parts.append(document)

                    else:
                        raise UnsupportedInputError(f"BinaryInput({inp.kind.value})", "anthropic")

                else:
                    raise UnsupportedInputError(type(inp).__name__, "anthropic")

            if content_parts:
                if len(content_parts) == 1 and (part := content_parts[0])["type"] == "text":
                    turns.append(("user", part["text"]))
                else:
                    turns.append(("user", content_parts))

        elif isinstance(message, CompactionSummary):
            # Surface the summary as a user turn so it stays visible and gives a valid opening turn
            turns.append(("user", f"[Summary of earlier conversation]\n{message.summary}"))

        elif isinstance(message, (ToolResultEvent, ToolErrorEvent)):
            # Fallback path — an individual result event without a
            # ToolResultsEvent wrapper. This happens when the wrapper
            # fails to persist (the exact failure mode that motivated
            # `resolved_tool_ids` above). Emit as its own user turn so
            # the conversation stays consistent.
            parent = getattr(message, "parent_id", None)
            if parent and parent in valid_tool_ids and parent not in emitted_result_ids:
                emitted_result_ids.add(parent)
                turns.append(("user", [_tool_result_block(message, serializer, toolset_names.get(parent))]))

    return [{"role": role, "content": content} for role, content in turns]


def inject_cache_breakpoint(messages: list[MessageParam]) -> None:
    """Mark the end of the last user turn as the end of the cacheable prefix."""
    for message in reversed(messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            return
        blocks = list(content)
        # A user turn built by `convert_messages` ends in a text, media or tool_result block;
        # the SDK's response models and the thinking blocks cannot carry a breakpoint.
        if (
            blocks
            and isinstance(last := blocks[-1], dict)
            and last["type"] != "thinking"
            and last["type"] != "redacted_thinking"
        ):
            last["cache_control"] = {"type": "ephemeral"}
        message["content"] = blocks
        return


def _count_or_absent(value: Any) -> float | None:
    """A count the provider supplied, or ``None`` where it supplied nothing.

    Presence decides rather than truthiness: ``0`` is a measurement, absence is the
    lack of one, and a consumer that cannot tell them apart cannot tell "no cache hit"
    from "cache never measured".
    """
    return None if value is None else float(value)


def normalize_usage(raw: dict[str, Any]) -> Usage:
    """Normalize Anthropic's native usage keys to standard format.

    ``server_tool_use`` (a count of tool *requests*), the ``service_tier`` and
    ``inference_geo`` labels and ``cache_creation``'s per-lifetime breakdown are
    deliberately unmapped: ``Usage`` carries whole-call token counts only.
    """
    details = raw.get("output_tokens_details") or {}
    prompt = float(raw.get("input_tokens", 0))
    completion = float(raw.get("output_tokens", 0))
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cache_creation_input_tokens=_count_or_absent(raw.get("cache_creation_input_tokens")),
        cache_read_input_tokens=_count_or_absent(raw.get("cache_read_input_tokens")),
        # Output tokens spent on internal reasoning — a decomposition of
        # ``output_tokens``, not an addition to it, matching the OpenAI mapper.
        thinking_tokens=_count_or_absent(details.get("thinking_tokens")),
    )
