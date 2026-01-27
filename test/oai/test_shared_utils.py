"""Unit tests for autogen.oai.shared_utils module."""

import pytest
from pydantic import BaseModel

from autogen.oai.shared_utils import normalize_pydantic_schema_to_dict


class SimpleModel(BaseModel):
    """Simple Pydantic model without nested references."""

    name: str
    age: int


class Step(BaseModel):
    """Nested model used in references."""

    explanation: str
    output: str


class MathReasoning(BaseModel):
    """Model with nested references."""

    steps: list[Step]
    final_answer: str


class Extra(BaseModel):
    """Model used in additionalProperties test."""

    notes: str


class Output(BaseModel):
    """Model with additionalProperties (dict[str, T])."""

    is_good: bool
    extra: dict[str, Extra]


class ContactInfo(BaseModel):
    """Deeply nested model."""

    email: str
    phone: str


class Person(BaseModel):
    """Model with nested ContactInfo."""

    name: str
    contact: ContactInfo


class Project(BaseModel):
    """Complex nested model."""

    title: str
    owner: Person


def test_normalize_pydantic_schema_simple():
    """Test normalize_pydantic_schema_to_dict with simple Pydantic model."""
    normalized = normalize_pydantic_schema_to_dict(SimpleModel)

    # Should not have $defs
    assert "$defs" not in normalized

    # Should have correct structure
    assert normalized["type"] == "object"
    assert "properties" in normalized
    assert "name" in normalized["properties"]
    assert "age" in normalized["properties"]
    assert normalized["properties"]["name"]["type"] == "string"
    assert normalized["properties"]["age"]["type"] == "integer"


def test_normalize_pydantic_schema_with_refs():
    """Test normalize_pydantic_schema_to_dict resolves $ref references."""
    normalized = normalize_pydantic_schema_to_dict(MathReasoning, for_genai_api=False)

    # Verify $defs is removed
    assert "$defs" not in normalized

    # Verify $ref references are resolved
    assert "properties" in normalized
    assert "steps" in normalized["properties"]
    steps_schema = normalized["properties"]["steps"]

    # The $ref should be resolved to the actual Step schema
    assert "$ref" not in steps_schema.get("items", {})
    assert "properties" in steps_schema.get("items", {})
    assert "explanation" in steps_schema["items"]["properties"]
    assert "output" in steps_schema["items"]["properties"]

    # Verify final_answer is present
    assert "final_answer" in normalized["properties"]
    assert normalized["properties"]["final_answer"]["type"] == "string"


def test_normalize_pydantic_schema_complex_nested():
    """Test normalize_pydantic_schema_to_dict with complex nested Pydantic model."""
    normalized = normalize_pydantic_schema_to_dict(Project, for_genai_api=False)

    # Should not have $defs
    assert "$defs" not in normalized

    # Should have resolved all references
    assert normalized["type"] == "object"
    assert "properties" in normalized

    # Check nested owner structure (should have resolved Person -> ContactInfo)
    owner_prop = normalized["properties"]["owner"]
    assert "$ref" not in str(owner_prop)
    assert owner_prop["type"] == "object"
    assert "contact" in owner_prop["properties"]

    # Check deeply nested contact structure
    contact_prop = owner_prop["properties"]["contact"]
    assert "$ref" not in str(contact_prop)
    assert contact_prop["type"] == "object"
    assert "email" in contact_prop["properties"]
    assert "phone" in contact_prop["properties"]


def test_normalize_pydantic_schema_with_dict_input():
    """Test normalize_pydantic_schema_to_dict with dict schema containing $refs."""
    dict_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"$ref": "#/$defs/Step"},
            },
            "final_answer": {"type": "string"},
        },
        "$defs": {
            "Step": {
                "type": "object",
                "properties": {
                    "explanation": {"type": "string"},
                    "output": {"type": "string"},
                },
            },
        },
    }

    normalized = normalize_pydantic_schema_to_dict(dict_schema, for_genai_api=False)

    # Should not have $defs
    assert "$defs" not in normalized

    # Should have resolved $ref
    steps_items = normalized["properties"]["steps"]["items"]
    assert "$ref" not in steps_items
    assert "properties" in steps_items
    assert "explanation" in steps_items["properties"]
    assert "output" in steps_items["properties"]


def test_normalize_pydantic_schema_with_additional_properties_genai():
    """Test normalize_pydantic_schema_to_dict converts additionalProperties for GenAI API."""
    normalized_genai = normalize_pydantic_schema_to_dict(Output, for_genai_api=True)

    # Verify $defs is removed
    assert "$defs" not in normalized_genai

    # Verify additionalProperties is converted to a regular property
    assert "properties" in normalized_genai
    assert "extra" in normalized_genai["properties"]
    extra_schema = normalized_genai["properties"]["extra"]

    # Should have properties (converted from additionalProperties)
    assert "properties" in extra_schema
    assert "additionalProperties" not in extra_schema

    # The converted property should be named "value" and contain the Extra schema
    assert "value" in extra_schema["properties"]
    value_schema = extra_schema["properties"]["value"]
    assert "properties" in value_schema
    assert "notes" in value_schema["properties"]


def test_normalize_pydantic_schema_with_additional_properties_vertexai():
    """Test normalize_pydantic_schema_to_dict keeps additionalProperties for Vertex AI."""
    normalized_vertexai = normalize_pydantic_schema_to_dict(Output, for_genai_api=False)

    # Verify $defs is removed
    assert "$defs" not in normalized_vertexai

    # For Vertex AI, additionalProperties might be kept (depending on implementation)
    # But $ref should still be resolved
    assert "properties" in normalized_vertexai
    assert "extra" in normalized_vertexai["properties"]
    extra_schema_vertexai = normalized_vertexai["properties"]["extra"]

    # $ref should be resolved
    assert "$ref" not in extra_schema_vertexai.get("additionalProperties", {})


def test_normalize_pydantic_schema_invalid_input():
    """Test normalize_pydantic_schema_to_dict with invalid input raises error."""
    # Test with string
    with pytest.raises(ValueError, match="Schema must be a Pydantic model class or dict"):
        normalize_pydantic_schema_to_dict("not a schema")

    # Test with integer
    with pytest.raises(ValueError, match="Schema must be a Pydantic model class or dict"):
        normalize_pydantic_schema_to_dict(123)

    # Test with list
    with pytest.raises(ValueError, match="Schema must be a Pydantic model class or dict"):
        normalize_pydantic_schema_to_dict([1, 2, 3])


def test_normalize_pydantic_schema_missing_ref_definition():
    """Test normalize_pydantic_schema_to_dict with missing $ref definition raises error."""
    dict_schema = {
        "type": "object",
        "properties": {
            "step": {"$ref": "#/$defs/Step"},
        },
        "$defs": {},  # Missing Step definition
    }

    with pytest.raises(ValueError, match="Definition 'Step' not found in \\$defs"):
        normalize_pydantic_schema_to_dict(dict_schema)


def test_normalize_pydantic_schema_unsupported_ref_format():
    """Test normalize_pydantic_schema_to_dict with unsupported $ref format raises error."""
    dict_schema = {
        "type": "object",
        "properties": {
            "step": {"$ref": "#/definitions/Step"},  # Wrong format
        },
        "$defs": {
            "Step": {"type": "object"},
        },
    }

    with pytest.raises(ValueError, match="Unsupported \\$ref format"):
        normalize_pydantic_schema_to_dict(dict_schema)


def test_normalize_pydantic_schema_additional_properties_with_existing_properties():
    """Test that additionalProperties is removed when object already has properties."""
    dict_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "additionalProperties": {"type": "string"},
    }

    normalized = normalize_pydantic_schema_to_dict(dict_schema, for_genai_api=True)

    # additionalProperties should be removed when properties exist
    assert "additionalProperties" not in normalized
    assert "properties" in normalized
    assert "name" in normalized["properties"]


def test_normalize_pydantic_schema_nested_additional_properties():
    """Test that nested additionalProperties are also converted."""
    dict_schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
    }

    normalized = normalize_pydantic_schema_to_dict(dict_schema, for_genai_api=True)

    # Outer additionalProperties should be converted
    outer_schema = normalized["properties"]["outer"]
    assert "properties" in outer_schema
    assert "additionalProperties" not in outer_schema
    assert "value" in outer_schema["properties"]

    # Inner additionalProperties should also be converted
    inner_schema = outer_schema["properties"]["value"]
    assert "properties" in inner_schema
    assert "additionalProperties" not in inner_schema
    assert "value" in inner_schema["properties"]


def test_normalize_pydantic_schema_with_anyof_oneof_allof():
    """Test that additionalProperties in anyOf/oneOf/allOf are converted."""
    dict_schema = {
        "type": "object",
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        ],
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        ],
        "allOf": [
            {
                "type": "object",
                "additionalProperties": {"type": "boolean"},
            },
        ],
    }

    normalized = normalize_pydantic_schema_to_dict(dict_schema, for_genai_api=True)

    # Check anyOf
    assert "anyOf" in normalized
    assert "properties" in normalized["anyOf"][0]
    assert "additionalProperties" not in normalized["anyOf"][0]

    # Check oneOf
    assert "oneOf" in normalized
    assert "properties" in normalized["oneOf"][0]
    assert "additionalProperties" not in normalized["oneOf"][0]

    # Check allOf
    assert "allOf" in normalized
    assert "properties" in normalized["allOf"][0]
    assert "additionalProperties" not in normalized["allOf"][0]


def test_normalize_pydantic_schema_with_array_items():
    """Test that additionalProperties in array items are converted."""
    dict_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
        },
    }

    normalized = normalize_pydantic_schema_to_dict(dict_schema, for_genai_api=True)

    # Items schema should have additionalProperties converted
    items_schema = normalized["properties"]["items"]["items"]
    assert "properties" in items_schema
    assert "additionalProperties" not in items_schema
    assert "value" in items_schema["properties"]
