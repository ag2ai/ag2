from pprint import pprint
from textwrap import dedent
from typing import Any, Literal, Protocol

from fast_depends import Provider
from fast_depends import dependency_provider as default_provider
from fast_depends.core import CallModel, build_call_model
from fast_depends.pydantic import PydanticSerializer
from pydantic import BaseModel, Field, HttpUrl

from autogen import ConversableAgent, LLMConfig
from autogen.tools import Tool as OldTool


class FunctionSchema(BaseModel):
    type: Literal["function"] = Field(default="function", init=False)

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSONSchema
    strict: bool = True

    def model_post_init(self, context: Any) -> None:
        self.parameters = {"additionalProperties": not self.strict} | self.parameters


class Format(BaseModel):
    type: Literal["grammar"] = "grammar"
    syntax: Literal["lark", "regex"] = "regex"
    definition: str


class CustomToolSchema(BaseModel):
    type: Literal["custom"] = Field(default="custom", init=False)

    name: str
    description: str = ""
    format: Format | None = None


class MCPToolSchema(BaseModel):
    type: Literal["mcp"] = Field(default="mcp", init=False)
    server_label: str
    require_approval: Literal["never"]

    # MCP server
    server_description: str | None = ""
    server_url: HttpUrl | None

    # Connector
    connector_id: str | None
    authorization: str | None


class Tool(Protocol):
    def get_schema(self) -> FunctionSchema: ...

    # input - ToolCalls[]
    # output - ToolResult
    def call(self, arguments: str) -> Any: ...


class FunctionTool(Tool):
    def __init__(
        self,
        schema: FunctionSchema,
        call_model: CallModel,
    ) -> None:
        self.__schema = schema
        self.__call_model = call_model

    def get_schema(self) -> FunctionSchema:
        return self.__schema


class new_tool:
    def __init__(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool = True,
        # FastDepends options
        dependency_provider: Provider | None = None,
    ) -> None:
        self.__schema = {
            k: v
            for k, v in {
                "name": name,
                "description": description,
                "parameters": parameters,
                "strict": strict,
            }.items()
            if v is not None
        }
        self.__provider = dependency_provider or default_provider

    def __call__(self, func: callable) -> FunctionTool:
        call_model = build_call_model(
            func,
            dependency_provider=self.__provider,
            serializer_cls=PydanticSerializer(),
        )

        schema = {
            k: v
            for k, v in {
                "name": func.__name__,
                "description": getattr(func, "__doc__", None),
                "parameters": call_model.serializer.model.model_json_schema(),
            }.items()
            if v is not None
        } | self.__schema

        return FunctionTool(
            schema=FunctionSchema(**schema),
            call_model=call_model,
        )


@new_tool("func")
def my_func(username: str) -> str:
    """My function."""
    print(username)
    return f"Tool called by {username}"


llm_config = LLMConfig({"model": "gpt-5"})


schema = my_func.get_schema()
tool = OldTool(
    name=schema.name,
    description=schema.description,
    func_or_tool=my_func.call,
)
tool._func_schema = schema.model_dump()

agent = ConversableAgent(
    name="agent",
    system_message=dedent("""
        You are a helpful assistant that helps manage and discuss proverbs.

        The user has a list of proverbs that you can help them manage.
        You have tools available to add, set, or retrieve proverbs from the list.

        When discussing proverbs, ALWAYS use the get_proverbs tool to see the current list before
        mentioning, updating, or discussing proverbs with the user.
    """).strip(),
    llm_config=llm_config,
    functions=[tool],
)

_, result = agent.generate_oai_reply([
    {
        "content": "Please, call `func` tool with `John` name and show me result",
        "role": "user",
    }
])

pprint(result)
# 'tool_calls': [{'function': {'arguments': '{"arguments": "John"}',
#                               'name': 'func'},
#                  'id': 'chatcmpl-tool-94a4830f435dcc00',
#                  'type': 'function'}]}

# https://developers.openai.com/api/docs/guides/function-calling
# TODO:
# - tool
# - tool choice
# - parallel tools execution
# - streaming tool arguments
# - MCP tool
#    - include / exclude mcp functions
# - builtin tools
# - tool search tool
# - code execution
# - shell tool
#    - local shell tool
#    - environment shell tool
# - web search
# - tools filtering
# - image generation tools
# - computer use tool
# - apply_patch tool  # for IDE
# - file search tools
# Code features
# - tools DI
# - guarded tools
# - HITL inside tool
# - playground directory / file
# Additional
# - skills
