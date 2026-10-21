# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal, TypeAlias

ToolChoiceMode: TypeAlias = Literal["auto", "none", "required"]

_MODE_STRICTNESS: dict[ToolChoiceMode, int] = {"auto": 0, "required": 1, "none": 2}


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """Which tools one LLM call may use, and whether it has to use one.

    Set it on a single call from ``BaseMiddleware.on_llm_call`` with
    ``call_next(events, context.with_tool_choice(...))``; see ADR 0025.

    ``mode`` is provider-neutral: ``"auto"`` lets the model decide,
    ``"required"`` makes it call a tool, and ``"none"`` stops it from calling
    any. Each client maps it to its provider's own setting.

    ``tools``, when set, keeps only the tools with these names in the request:
    a function tool by its name, a builtin tool by its schema type (for
    example ``"web_search"``). ``None`` keeps every tool. A choice that leaves
    no tool to call is ``"none"``.
    """

    mode: ToolChoiceMode = "auto"
    tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.mode not in _MODE_STRICTNESS:
            raise ValueError(f"Unknown tool choice mode {self.mode!r}; expected one of {tuple(_MODE_STRICTNESS)}.")

        tools = None if self.tools is None else frozenset(self.tools)
        if self.mode == "none" or tools == frozenset():
            # Nothing is callable either way, and "none" keeps the schemas in the
            # request: some providers reject a tool history sent without them.
            object.__setattr__(self, "mode", "none")
            tools = None
        object.__setattr__(self, "tools", tools)

    @classmethod
    def only(cls, *names: str, mode: ToolChoiceMode = "auto") -> "ToolChoice":
        """Allow only the tools named here."""
        return cls(mode, frozenset(names))

    @classmethod
    def ensure(cls, choice: "ToolChoice | ToolChoiceMode") -> "ToolChoice":
        if isinstance(choice, ToolChoice):
            return choice
        return cls(choice)

    def narrow(self, inner: "ToolChoice | ToolChoiceMode") -> "ToolChoice":
        """Combine this choice with one made closer to the client.

        The result never allows more than either: the allowed tools are the
        intersection, and the stricter mode wins (``"none"`` over
        ``"required"`` over ``"auto"``).
        """
        inner = ToolChoice.ensure(inner)

        if self.tools is None:
            tools = inner.tools
        elif inner.tools is None:
            tools = self.tools
        else:
            tools = self.tools & inner.tools

        mode = max(self.mode, inner.mode, key=_MODE_STRICTNESS.__getitem__)
        return ToolChoice(mode, tools)

    def allows(self, name: str) -> bool:
        return self.mode != "none" and (self.tools is None or name in self.tools)
