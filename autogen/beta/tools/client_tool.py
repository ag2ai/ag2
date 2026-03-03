from contextlib import ExitStack
from typing import Any

from autogen.beta.annotations import Context
from autogen.beta.events import ClientToolCall, ToolCall

from .schemas import FunctionToolSchema
from .tool import Tool


class ClientTool(Tool):
    __slots__ = (
        "schema",
        "name",
    )

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = FunctionToolSchema.model_validate(schema)
        self.name = self.schema.function.name

    def register(self, stack: "ExitStack", ctx: "Context") -> None:
        stack.enter_context(
            ctx.stream.where((ToolCall.name == self.name) & ClientToolCall.not_()).sub_scope(self.execute),
        )

    async def execute(self, event: "ToolCall", ctx: "Context") -> None:
        ev = ClientToolCall.from_call(event)
        await ctx.send(ev)
