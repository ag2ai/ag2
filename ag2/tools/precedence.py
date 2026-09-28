# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable, Iterator
from copy import copy
from dataclasses import dataclass

from ag2.annotations import Context

from .builtin.tool_search import ToolSearchTool
from .final import FunctionToolSchema, Toolkit
from .schemas import ToolSchema
from .tool import Tool

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedTools:
    """The tools a turn registers and the schemas it exposes, one tool per callable name."""

    tools: list[Tool]
    schemas: list[ToolSchema]
    known_tools: set[str]


@dataclass(slots=True, eq=False)
class _Leaf:
    tool: Tool
    keys: list[tuple[str, bool]]
    """``(name, is_function)`` for each schema: function name, or built-in type."""
    source: str


async def resolve_tools(tools: Iterable[Tool], context: Context) -> ResolvedTools:
    """Resolve ``tools`` so that every name the model can call belongs to one tool.

    The model calls a tool by name and a ``ToolCallEvent`` reaches every tool
    subscribed to that name, so a turn keeps exactly one tool per name. Toolkits
    are resolved member by member, after they discover their tools.

    Precedence, when two tools share a name:

    * Tools declared in code override each other in order: the later one wins,
      like a dict update, with a warning. Built-in tools count as declared in code.
    * A tool a remote peer provides at runtime (an MCP server's tool, a tool a
      client sends) never overrides one declared in code, wherever it sits; it
      is dropped with a warning.
    * Between two remote peers, the first one's tool wins; the other is
      dropped with a warning.

    Built-in schemas of one type may repeat (e.g. several ``MCPServerTool``
    servers); only a function name clashes with them. A dropped tool is not
    registered, so it never receives a call.
    """
    declared = list(tools)
    leaves: list[_Leaf] = []
    for tool in declared:
        await _collect_leaves(tool, context, "", leaves)

    keep = iter(_select(leaves))
    resolved = [pruned for tool in declared if (pruned := _prune(tool, keep)) is not None]

    schemas = [schema for tool in resolved for schema in await tool.schemas(context)]
    known_tools = {s.function.name if isinstance(s, FunctionToolSchema) else s.type for s in schemas}
    return ResolvedTools(resolved, schemas, known_tools)


async def _collect_leaves(tool: Tool, context: Context, parent: str, out: list[_Leaf]) -> None:
    source = f"{parent}{type(tool).__name__}({tool.name!r})"
    if isinstance(tool, Toolkit | ToolSearchTool):
        # A toolkit may discover its members here (e.g. ``MCPToolkit``).
        await tool.schemas(context)
        for child in tool.tools:
            await _collect_leaves(child, context, f"{source} > ", out)
        return

    keys = [
        (s.function.name, True) if isinstance(s, FunctionToolSchema) else (s.type, False)
        for s in await tool.schemas(context)
    ]
    out.append(_Leaf(tool, keys, source))


def _select(leaves: list[_Leaf]) -> list[bool]:
    """Whether each leaf, in order, keeps its place in the turn."""
    owners: dict[str, list[tuple[_Leaf, bool]]] = {}
    kept: list[_Leaf] = []
    for leaf in leaves:
        rivals: list[_Leaf] = []
        for name, is_function in leaf.keys:
            for rival, rival_is_function in owners.get(name, ()):
                if (is_function or rival_is_function) and rival not in rivals:
                    rivals.append(rival)

        # A tool declared in code wins over every rival; a tool a remote peer
        # provides loses to any rival already in place.
        if rivals and not leaf.tool.declared_in_code:
            _report(rivals[0], leaf)
            continue

        for rival in rivals:
            _report(leaf, rival)
            kept.remove(rival)
            for entries in owners.values():
                entries[:] = [entry for entry in entries if entry[0] is not rival]

        kept.append(leaf)
        for name, is_function in leaf.keys:
            owners.setdefault(name, []).append((leaf, is_function))

    return [leaf in kept for leaf in leaves]


def _report(winner: _Leaf, loser: _Leaf) -> None:
    name = next(n for n, _ in loser.keys if any(n == w for w, _ in winner.keys))
    if loser.tool.declared_in_code:
        logger.warning("Tool `%s` from %s is overridden by %s.", name, loser.source, winner.source)
    elif winner.tool.declared_in_code:
        logger.warning(
            "Tool `%s` provided by %s is ignored: %s declares a tool with that name, "
            "and tools declared in code take precedence over tools a remote peer provides.",
            name,
            loser.source,
            winner.source,
        )
    else:
        logger.warning(
            "Tool `%s` provided by %s is ignored: %s already provides it. "
            "If the tools come from MCP servers, set `tool_name_prefix` on a server config to keep both.",
            name,
            loser.source,
            winner.source,
        )


def _prune(tool: Tool, keep: Iterator[bool]) -> Tool | None:
    """``tool`` without the members dropped by :func:`_select`; ``None`` if it is dropped itself.

    Walks members in the order :func:`_collect_leaves` did, consuming one flag per leaf.
    """
    if not isinstance(tool, Toolkit | ToolSearchTool):
        return tool if next(keep) else None

    members = {name: _prune(child, keep) for name, child in tool._tools.items()}
    if all(member is child for member, child in zip(members.values(), tool.tools, strict=True)):
        return tool

    pruned = copy(tool)
    pruned._tools = {name: member for name, member in members.items() if member is not None}
    return pruned
