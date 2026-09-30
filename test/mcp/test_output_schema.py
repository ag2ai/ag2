# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from dirty_equals import IsPartialDict
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from pydantic import (
    AliasChoices,
    AliasGenerator,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    GetJsonSchemaHandler,
    Json,
    RootModel,
    computed_field,
    field_serializer,
    model_serializer,
    with_config,
)
from pydantic.alias_generators import to_camel, to_pascal
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypedDict

from ag2 import Agent
from ag2.events import ModelRequest
from ag2.mcp import MCPServer, build_ask_tool, mcp_tool
from ag2.mcp.testing import connect, connect_modern
from ag2.response import PromptedSchema, ResponseSchema, response_schema
from ag2.testing import TestConfig, TrackingConfig

from ._helpers import tool_named


class AliasedItem(BaseModel):
    item_id: str = Field(alias="itemId")


class ValidationAliasItem(BaseModel):
    item_id: str = Field(validation_alias="inputId")


class SerializationAliasItem(BaseModel):
    item_id: str = Field(serialization_alias="outputId")


class SplitAliasesItem(BaseModel):
    item_id: str = Field(validation_alias="inputId", serialization_alias="outputId")


class ChoiceAliasItem(BaseModel):
    item_id: str = Field(validation_alias=AliasChoices("itemId", "item_id"))


class PathAliasItem(BaseModel):
    item_id: str = Field(validation_alias=AliasPath("payload", "itemId"))


class ConfiguredItem(SplitAliasesItem):
    model_config = ConfigDict(serialize_by_alias=True)


class GeneratedAliasesItem(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(validation_alias=to_camel, serialization_alias=to_pascal),
        serialize_by_alias=True,
    )
    item_id: str


class NestedItems(BaseModel):
    items: list[AliasedItem]


class ItemMap(RootModel[dict[str, AliasedItem]]):
    pass


class RecursiveItem(AliasedItem):
    children: list["RecursiveItem"] = Field(default_factory=list)


class ItemTree(BaseModel):
    root: RecursiveItem


class DefaultParent(BaseModel):
    child: ConfiguredItem
    label: str = Field(serialization_alias="wireLabel")


class AliasedParent(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)
    child: AliasedItem = Field(alias="entry")
    configured: ConfiguredItem


@dataclass
class DataclassItem:
    item_id: str = Field(alias="itemId")


class TypedDictItem(TypedDict):
    item_id: Annotated[str, Field(alias="itemId")]


@with_config(ConfigDict(serialize_by_alias=False))
@dataclass
class ConfiguredDataclassItem(DataclassItem):
    pass


@with_config(ConfigDict(serialize_by_alias=False))
class ConfiguredTypedDictItem(TypedDictItem):
    pass


class NestedContainers(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)
    dataclass_item: DataclassItem
    typed_dict_item: TypedDictItem
    configured_dataclass: ConfiguredDataclassItem
    configured_typed_dict: ConfiguredTypedDictItem


class JsonPayload(BaseModel):
    payload: Json[dict[str, int]]


class SerializedFields(BaseModel):
    value: int
    hidden: str = Field(exclude=True)

    @field_serializer("value")
    def stringify(self, value: int) -> str:
        return str(value)

    @computed_field(alias="doubleValue")
    @property
    def doubled(self) -> int:
        return self.value * 2


class SerializedModel(BaseModel):
    name: str

    @model_serializer
    def summarize(self) -> dict[str, str]:
        return {"summary": self.name}


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", [connect, connect_modern], ids=["handshake", "modern"])
@pytest.mark.parametrize("tool_name", ["ask", "report"])
class TestModelOutputSchema:
    @pytest.mark.parametrize(
        ("model", "input_data", "expected"),
        [
            pytest.param(AliasedItem, {"itemId": "a"}, {"item_id": "a"}, id="alias"),
            pytest.param(ValidationAliasItem, {"inputId": "a"}, {"item_id": "a"}, id="validation-alias"),
            pytest.param(SerializationAliasItem, {"item_id": "a"}, {"item_id": "a"}, id="serialization-alias"),
            pytest.param(SplitAliasesItem, {"inputId": "a"}, {"item_id": "a"}, id="split-aliases"),
            pytest.param(ChoiceAliasItem, {"itemId": "a"}, {"item_id": "a"}, id="alias-choices"),
            pytest.param(PathAliasItem, {"payload": {"itemId": "a"}}, {"item_id": "a"}, id="alias-path"),
            pytest.param(ConfiguredItem, {"inputId": "a"}, {"outputId": "a"}, id="serialize-by-alias"),
            pytest.param(GeneratedAliasesItem, {"itemId": "a"}, {"ItemId": "a"}, id="alias-generator"),
            pytest.param(NestedItems, {"items": [{"itemId": "a"}]}, {"items": [{"item_id": "a"}]}, id="nested-list"),
            pytest.param(ItemMap, {"a": {"itemId": "a"}}, {"a": {"item_id": "a"}}, id="root-model"),
            pytest.param(
                ItemTree,
                {"root": {"itemId": "a", "children": [{"itemId": "b"}]}},
                {"root": {"item_id": "a", "children": [{"item_id": "b", "children": []}]}},
                id="recursive-reference",
            ),
            pytest.param(
                DefaultParent,
                {"child": {"inputId": "a"}, "label": "group"},
                {"child": {"outputId": "a"}, "label": "group"},
                id="child-enables-aliases",
            ),
            pytest.param(
                AliasedParent,
                {"entry": {"itemId": "a"}, "configured": {"inputId": "b"}},
                {"entry": {"item_id": "a"}, "configured": {"outputId": "b"}},
                id="child-keeps-field-names",
            ),
            pytest.param(
                NestedContainers,
                {
                    "dataclass_item": {"itemId": "a"},
                    "typed_dict_item": {"itemId": "b"},
                    "configured_dataclass": {"itemId": "c"},
                    "configured_typed_dict": {"itemId": "d"},
                },
                {
                    "dataclass_item": {"itemId": "a"},
                    "typed_dict_item": {"itemId": "b"},
                    "configured_dataclass": {"item_id": "c"},
                    "configured_typed_dict": {"itemId": "d"},
                },
                id="nested-dataclass-and-typed-dict",
            ),
            pytest.param(JsonPayload, {"payload": '{"count": 3}'}, {"payload": {"count": 3}}, id="json-field"),
            pytest.param(
                SerializedFields,
                {"value": 7, "hidden": "private"},
                {"value": "7", "doubled": 14},
                id="serialized-excluded-and-computed-fields",
            ),
            pytest.param(SerializedModel, {"name": "a"}, {"summary": "a"}, id="model-serializer"),
        ],
    )
    async def test_schema_matches_unchanged_model_dump(
        self,
        connector: Callable[..., AbstractAsyncContextManager[ClientSession]],
        tool_name: str,
        model: type[BaseModel],
        input_data: dict[str, Any],
        expected: dict[str, Any],
    ) -> None:
        response_schema = ResponseSchema(model)
        validation_schema = deepcopy(response_schema.json_schema)
        agent = Agent("reporter", config=TestConfig(json.dumps(input_data)), response_schema=response_schema)

        @mcp_tool
        def report() -> model:
            return model.model_validate(input_data)

        async with connector(MCPServer(agent, tools=[report], sessions=False)) as session:
            listed = tool_named(await session.list_tools(), tool_name)
            # The SDK validates structuredContent against this advertised schema.
            result = await session.call_tool(tool_name, {"message": "report"} if tool_name == "ask" else {})

        assert listed.output_schema is not None
        assert result.is_error is False
        assert result.structured_content == expected
        assert response_schema.json_schema == validation_schema

    @pytest.mark.parametrize("serialize_by_alias", [False, True])
    async def test_required_nullable_fields_remain_present(
        self,
        connector: Callable[..., AbstractAsyncContextManager[ClientSession]],
        tool_name: str,
        serialize_by_alias: bool,
    ) -> None:
        class NullableReport(BaseModel):
            model_config = ConfigDict(serialize_by_alias=serialize_by_alias)
            value: int | None = Field(validation_alias="inputValue", serialization_alias="outputValue")

        input_data = {"inputValue": None}
        agent = Agent("reporter", config=TestConfig(json.dumps(input_data)), response_schema=NullableReport)

        @mcp_tool
        def report() -> NullableReport:
            return NullableReport.model_validate(input_data)

        async with connector(MCPServer(agent, tools=[report], sessions=False)) as session:
            listed = tool_named(await session.list_tools(), tool_name)
            result = await session.call_tool(tool_name, {"message": "report"} if tool_name == "ask" else {})

        output_key = "outputValue" if serialize_by_alias else "value"
        assert listed.output_schema == IsPartialDict({
            "properties": {output_key: IsPartialDict({"anyOf": [{"type": "integer"}, {"type": "null"}]})},
            "required": [output_key],
        })
        assert result.is_error is False
        assert result.structured_content == {output_key: None}

    async def test_alias_schema_describes_output_keys(
        self,
        connector: Callable[..., AbstractAsyncContextManager[ClientSession]],
        tool_name: str,
    ) -> None:
        agent = Agent("reporter", config=TestConfig('{"itemId": "a"}'), response_schema=AliasedItem)

        @mcp_tool
        def report() -> AliasedItem:
            return AliasedItem(itemId="a")

        async with connector(MCPServer(agent, tools=[report])) as session:
            listed = tool_named(await session.list_tools(), tool_name)

        assert listed.output_schema == {
            "type": "object",
            "properties": {"item_id": {"title": "Item Id", "type": "string"}},
            "required": ["item_id"],
        }
        assert agent.response_schema.json_schema == IsPartialDict({
            "properties": {"itemId": {"title": "Itemid", "type": "string"}},
            "required": ["itemId"],
        })


@pytest.mark.asyncio
@pytest.mark.parametrize("connector", [connect, connect_modern], ids=["handshake", "modern"])
class TestOutputSchemaCompatibility:
    async def test_serialized_field_names_do_not_relax_llm_validation(
        self, connector: Callable[..., AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        config = TrackingConfig(TestConfig('{"item_id": "a"}'))
        agent = Agent("reporter", config=config, response_schema=AliasedItem)

        async with connector(MCPServer(agent, sessions=False)) as session:
            result = await session.call_tool("ask", {"message": "report"})

        assert result.is_error is True
        assert result.structured_content is None
        config.mock.assert_called_once_with(ModelRequest.ensure_request(["report"]))

    async def test_explicit_output_schema_and_manual_result_are_untouched(
        self, connector: Callable[..., AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        declared = {
            "type": "object",
            "properties": {"manualId": {"type": "string", "const": "a"}},
            "required": ["manualId"],
            "additionalProperties": False,
        }
        expected = CallToolResult(
            content=[TextContent(type="text", text="custom text")],
            structuredContent={"manualId": "a"},
            _meta={"custom/key": "kept"},
        )

        @mcp_tool(output_schema=declared)
        def report() -> AliasedItem:
            # A complete result may be returned even under a model annotation.
            return expected  # type: ignore[return-value]

        agent = Agent("reporter", config=TestConfig("unused"))
        async with connector(MCPServer(agent, tools=[report])) as session:
            listed = tool_named(await session.list_tools(), "report")
            result = await session.call_tool("report", {})

        assert listed.output_schema == declared
        assert result.content == expected.content
        assert result.structured_content == expected.structured_content
        assert result.meta == IsPartialDict({"custom/key": "kept"})
        assert result.is_error is False

    async def test_custom_response_proto_keeps_its_declared_schema(
        self, connector: Callable[..., AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        declared = {
            "type": "object",
            "properties": {"customId": {"type": "string"}},
            "required": ["customId"],
            "description": "A user-owned response contract.",
        }

        @response_schema(schema=declared, embed=False)
        def parse(content: dict[str, str]) -> dict[str, str]:
            return content

        agent = Agent("reporter", config=TestConfig('{"customId": "a"}'), response_schema=parse)
        async with connector(MCPServer(agent, sessions=False)) as session:
            listed = tool_named(await session.list_tools(), "ask")
            result = await session.call_tool("ask", {"message": "report"})

        assert listed.output_schema == declared
        assert result.is_error is False
        assert result.structured_content == {"customId": "a"}

    async def test_prompted_schema_still_returns_only_text(
        self, connector: Callable[..., AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        raw = '{"itemId": "a"}'
        agent = Agent("reporter", config=TestConfig(raw), response_schema=PromptedSchema(AliasedItem))
        async with connector(MCPServer(agent, sessions=False)) as session:
            listed = tool_named(await session.list_tools(), "ask")
            result = await session.call_tool("ask", {"message": "report"})

        assert listed.output_schema is None
        assert result.content == [TextContent(type="text", text=raw)]
        assert result.structured_content is None
        assert result.is_error is False

    async def test_repeated_listings_and_calls_do_not_regenerate_schemas(
        self, connector: Callable[..., AbstractAsyncContextManager[ClientSession]]
    ) -> None:
        generated: list[str] = []

        class Report(AliasedItem):
            @classmethod
            def __get_pydantic_json_schema__(
                cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
            ) -> JsonSchemaValue:
                generated.append(handler.mode)
                return handler(schema)

        agent = Agent("reporter", config=TestConfig('{"itemId": "a"}'), response_schema=Report)
        server = MCPServer(agent, sessions=False)
        prepared = list(generated)

        async with connector(server) as session:
            for _ in range(2):
                listed = tool_named(await session.list_tools(), "ask")
                result = await session.call_tool("ask", {"message": "report"})
                assert listed.output_schema is not None
                assert result.structured_content == {"item_id": "a"}

        assert generated == prepared
        assert generated.count("serialization") == 1


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_response_schema_root_metadata_is_preserved(explicit_metadata: bool) -> None:
    class DescribedItem(AliasedItem):
        """An item with presentation metadata."""

    schema = ResponseSchema(
        DescribedItem,
        name="custom" if explicit_metadata else None,
        description="custom description" if explicit_metadata else None,
    )
    expected = {
        "type": "object",
        "properties": {"item_id": {"title": "Item Id", "type": "string"}},
        "required": ["item_id"],
    }
    if explicit_metadata:
        expected.update(title="DescribedItem", description="An item with presentation metadata.")

    listed = build_ask_tool(Agent("reporter"), response_schema=schema)

    assert listed.output_schema == expected
