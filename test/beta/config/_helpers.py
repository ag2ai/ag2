# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from autogen.beta.tools.final import FunctionDefinition, FunctionToolSchema


@dataclass
class ToolStub:
    name: str
    schema: FunctionToolSchema


def make_parameterless_tool() -> ToolStub:
    return ToolStub(
        name="ask_human",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="ask_human",
                description="Ask the human for input.",
                parameters={"type": "null"},
            )
        ),
    )


def make_tool() -> ToolStub:
    return ToolStub(
        name="search_docs",
        schema=FunctionToolSchema(
            function=FunctionDefinition(
                name="search_docs",
                description="Search documentation by query.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["query"],
                },
            )
        ),
    )
