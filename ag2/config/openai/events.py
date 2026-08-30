# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from base64 import b64decode
from typing import Any, TypeAlias

from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionWebSearch,
    ResponseReasoningItem,
)
from openai.types.responses.response_code_interpreter_tool_call import OutputImage, OutputLogs
from openai.types.responses.response_function_web_search import ActionFind, ActionOpenPage, ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall, McpListTools

from ag2.events import (
    BaseEvent,
    BinaryInput,
    BinaryType,
    BuiltinToolCallEvent,
    BuiltinToolResultEvent,
    Field,
    Input,
    ModelReasoning,
    ProviderReplay,
    TextInput,
    ToolResult,
    UrlInput,
)
from ag2.tools.builtin.code_execution import CODE_EXECUTION_TOOL_NAME
from ag2.tools.builtin.file_search import FILE_SEARCH_TOOL_NAME
from ag2.tools.builtin.image_generation import IMAGE_GENERATION_TOOL_NAME
from ag2.tools.builtin.mcp_server import MCP_SERVER_TOOL_NAME
from ag2.tools.builtin.shell import SHELL_TOOL_NAME
from ag2.tools.builtin.web_search import WEB_SEARCH_TOOL_NAME

OpenAIServerToolItem: TypeAlias = (
    ResponseFunctionWebSearch
    | ResponseCodeInterpreterToolCall
    | ImageGenerationCall
    | ResponseFileSearchToolCall
    | McpCall
    | McpListTools
    | ResponseFunctionShellToolCall
)


class OpenAIServerToolCallEvent(BuiltinToolCallEvent):
    item: OpenAIServerToolItem = Field(repr=False)

    @classmethod
    def from_item(cls, item: object) -> "OpenAIServerToolCallEvent | None":
        if isinstance(item, ResponseFunctionWebSearch):
            return cls(
                id=item.id,
                name=WEB_SEARCH_TOOL_NAME,
                # warnings=False: pydantic 2.x warns on Action discriminated-union
                # serialization and on action.sources[].type values that the SDK
                # has not caught up to (e.g. "api"). The warning is informational —
                # the dump still produces correct JSON for round-trip.
                arguments=item.action.model_dump_json(warnings=False),
                item=item,
            )
        if isinstance(item, ResponseCodeInterpreterToolCall):
            return cls(
                id=item.id,
                name=CODE_EXECUTION_TOOL_NAME,
                arguments=json.dumps({"code": item.code}) if item.code is not None else "{}",
                item=item,
            )
        if isinstance(item, ImageGenerationCall) and item.result:
            return cls(
                id=item.id,
                name=IMAGE_GENERATION_TOOL_NAME,
                arguments="",
                item=item,
            )
        if isinstance(item, ResponseFileSearchToolCall):
            return cls(
                id=item.id,
                name=FILE_SEARCH_TOOL_NAME,
                arguments=json.dumps({"queries": item.queries}),
                item=item,
            )
        if isinstance(item, McpCall):
            return cls(
                id=item.id,
                name=MCP_SERVER_TOOL_NAME,
                arguments=item.arguments,
                item=item,
            )
        if isinstance(item, McpListTools):
            # A listing is not a tool invocation, but it is the only place a
            # server that could not be reached shows up at all. Reporting it as a
            # call keeps the failure observable instead of silently absent.
            return cls(
                id=item.id,
                name=MCP_SERVER_TOOL_NAME,
                arguments=json.dumps({"server_label": item.server_label}),
                item=item,
            )
        if isinstance(item, ResponseFunctionShellToolCall):
            return cls(
                id=item.id,
                name=SHELL_TOOL_NAME,
                arguments=json.dumps({"commands": list(item.action.commands)}),
                item=item,
            )
        return None


class OpenAIServerToolResultEvent(BuiltinToolResultEvent):
    item: ResponseFunctionShellToolCallOutput | None = Field(default=None, repr=False)
    """The output item a hosted shell call's result was built from.

    Every other hosted tool carries its outcome on the call item itself, which the
    call event already replays. A shell call does not: the container's output is a
    separate ``shell_call_output`` item, and replaying the call without it sends
    the API a command with no outcome. So this result carries it.
    """

    @classmethod
    def from_item(cls, item: object, *, parent_id: str) -> "OpenAIServerToolResultEvent | None":
        name: str
        parts: list[Input] = []
        metadata: dict[str, Any] = {}

        if isinstance(item, ResponseFunctionWebSearch):
            name = WEB_SEARCH_TOOL_NAME
            action = item.action
            metadata = {"action_type": action.type, "status": item.status}
            if isinstance(action, ActionSearch):
                # `sources` is populated only when the request asks for it via
                # include=["web_search_call.action.sources"]. The SDK declares
                # source.url as `str`, but the API has been observed to return
                # entries with empty url for synthesised/internal sources —
                # skip them rather than emit UrlInput(None).
                for source in action.sources or []:
                    if source.url:
                        parts.append(UrlInput(source.url, kind=BinaryType.BINARY))
                if action.queries:
                    metadata["queries"] = list(action.queries)
            elif isinstance(action, ActionOpenPage):
                if action.url:
                    parts.append(UrlInput(action.url, kind=BinaryType.BINARY))
            elif isinstance(action, ActionFind):
                parts.append(UrlInput(action.url, kind=BinaryType.BINARY))
                metadata["pattern"] = action.pattern

        elif isinstance(item, ResponseCodeInterpreterToolCall):
            name = CODE_EXECUTION_TOOL_NAME
            for output in item.outputs or []:
                if isinstance(output, OutputLogs):
                    parts.append(TextInput(output.logs))
                elif isinstance(output, OutputImage):
                    parts.append(UrlInput(output.url, kind=BinaryType.IMAGE))
            metadata = {"container_id": item.container_id, "status": item.status}

        elif isinstance(item, ImageGenerationCall) and item.result:
            name = IMAGE_GENERATION_TOOL_NAME
            parts = [BinaryInput(b64decode(item.result), media_type="image/png", kind=BinaryType.IMAGE)]
            metadata = item.model_dump(exclude={"result", "status", "type"})

        elif isinstance(item, ResponseFileSearchToolCall):
            name = FILE_SEARCH_TOOL_NAME
            metadata = {"status": item.status}
            results_meta: list[dict[str, Any]] = []
            for r in item.results or []:
                # `text` is populated only when the request asked for it via
                # include=["file_search_call.results"].
                if r.text:
                    parts.append(TextInput(r.text))
                results_meta.append({"file_id": r.file_id, "filename": r.filename, "score": r.score})
            if results_meta:
                metadata["results"] = results_meta

        elif isinstance(item, McpCall):
            name = MCP_SERVER_TOOL_NAME
            metadata = {"server_label": item.server_label, "tool": item.name, "status": item.status}
            if item.output:
                parts.append(TextInput(item.output))
            if item.error is not None:
                # A discriminated union: a protocol error and an HTTP error each
                # carry a code and a message, a tool execution error carries the
                # tool's own content. Dumping it whole keeps `type` — the
                # discriminator — next to that arm's own fields, so a caller can
                # branch instead of matching on prose.
                metadata["error"] = item.error.model_dump(mode="json")

        elif isinstance(item, McpListTools):
            name = MCP_SERVER_TOOL_NAME
            metadata = {"server_label": item.server_label, "tools": [t.name for t in item.tools]}
            if item.error is not None:
                metadata["error"] = item.error

        elif isinstance(item, ResponseFunctionShellToolCallOutput):
            # Paired with its `shell_call` by `ShellCallTracker`, which supplies
            # both the parent id and the commands through `from_shell_output`.
            return None

        else:
            return None

        return cls(parent_id=parent_id, name=name, result=ToolResult(parts=parts, metadata=metadata))

    @classmethod
    def from_shell_output(
        cls,
        item: ResponseFunctionShellToolCallOutput,
        *,
        call: ResponseFunctionShellToolCall,
        parent_id: str,
    ) -> "OpenAIServerToolResultEvent":
        """Build the result of a hosted shell call from the output item answering it."""
        parts: list[Input] = []
        outputs: list[dict[str, Any]] = []

        for output in item.output:
            if output.stdout:
                parts.append(TextInput(output.stdout))
            if output.stderr:
                parts.append(TextInput(output.stderr))
            outputs.append({
                "stdout": output.stdout,
                "stderr": output.stderr,
                "outcome": output.outcome.model_dump(),
            })

        return cls(
            parent_id=parent_id,
            name=SHELL_TOOL_NAME,
            item=item,
            result=ToolResult(
                parts=parts,
                metadata={
                    "commands": list(call.action.commands),
                    "status": item.status,
                    "outputs": outputs,
                },
            ),
        )


class ShellCallTracker:
    """Pairs a hosted ``shell_call`` with the ``shell_call_output`` answering it.

    The Responses API delivers a hosted shell call as two output items linked by
    ``call_id``: the command the model composed, then the container's output. ag2
    reports the pair as one call event and one result event, so the result has to
    reach back for the command that produced it — which nothing in the output item
    itself carries.

    A call with no output yet still produces its call event; only the result waits.
    """

    __slots__ = ("_open",)

    def __init__(self) -> None:
        self._open: dict[str, tuple[str, ResponseFunctionShellToolCall]] = {}

    def opened(self, call: ResponseFunctionShellToolCall, *, event_id: str) -> None:
        self._open[call.call_id] = (event_id, call)

    def close(self, item: ResponseFunctionShellToolCallOutput) -> OpenAIServerToolResultEvent | None:
        """Return the result event for ``item``, or ``None`` if its call was never seen."""
        opened = self._open.pop(item.call_id, None)
        if opened is None:
            return None

        parent_id, call = opened
        return OpenAIServerToolResultEvent.from_shell_output(item, call=call, parent_id=parent_id)


class OpenAIShellCommandChunk(BaseEvent):
    """A slice of the shell command the model is composing.

    Transient: superseded by the finished ``shell_call``, whose
    :class:`OpenAIServerToolCallEvent` carries every command in full.

    Kept apart from :class:`~ag2.events.ModelMessageChunk` deliberately — that
    chunk is the assistant's reply to the user, and mixing a command into it
    corrupts the reply for anyone concatenating chunks into an answer.
    """

    __transient__ = True

    content: str = Field(kw_only=False)
    command_index: int
    """Which command in the call this slice belongs to."""

    output_index: int


class OpenAIShellOutputChunk(BaseEvent):
    """A slice of the output a container produced running a shell command.

    Separate from :class:`OpenAIShellCommandChunk` rather than one type with a
    discriminator: this half carries standard output and standard error, and a
    single type would have to make both optional to accommodate the other half,
    leaving callers to branch on emptiness.

    Transient, for the same reason as the command chunk.
    """

    __transient__ = True

    command_index: int
    """Which command in the call produced this output."""

    output_index: int
    item_id: str
    stdout: str | None = None
    stderr: str | None = None


class OpenAIReasoningEvent(ModelReasoning, ProviderReplay):
    """Reasoning item the Responses API pairs with a server-side tool call.

    ProviderReplay anchor: the API rejects a replayed ``web_search_call`` whose
    ``reasoning`` item is missing. Persisted, unlike ``ModelReasoning``.
    """

    __transient__ = False
    __replay_role__ = "anchor"

    item: ResponseReasoningItem = Field(repr=False)
