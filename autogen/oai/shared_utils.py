from typing import Any

from pydantic import BaseModel


def normalize_pydantic_schema_to_dict(
        self, schema: dict[str, Any] | type[BaseModel], for_genai_api: bool = False
    ) -> dict[str, Any]:
        """
        Convert a Pydantic model's JSON schema to a flat dict schema by resolving $ref references.

        Similar to bedrock.py's _normalize_pydantic_schema_to_dict, but also handles
        additionalProperties conversion for Gemini GenAI API compatibility.

        Args:
            schema: Either a Pydantic model class or a dict containing the JSON schema
            for_genai_api: If True, convert additionalProperties to regular properties
                          for Gemini GenAI API compatibility (not needed for Vertex AI)

        Returns:
            A normalized dict schema with all $ref references resolved, $defs removed,
            and additionalProperties converted to properties if for_genai_api is True
        """
        from pydantic import BaseModel

        # If it's a Pydantic model, get its JSON schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            schema_dict = schema.model_json_schema()
        elif isinstance(schema, dict):
            schema_dict = schema.copy()
        else:
            raise ValueError(f"Schema must be a Pydantic model class or dict, got {type(schema)}")

        # Extract $defs if present
        defs = schema_dict.get("$defs", {}).copy()

        def resolve_ref(ref: str, definitions: dict[str, Any]) -> dict[str, Any]:
            """Resolve a $ref to its actual schema definition."""
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported $ref format: {ref}. Only '#/$defs/...' is supported.")
            # Extract the definition name from "#/$defs/Name"
            def_name = ref.split("/")[-1]
            if def_name not in definitions:
                raise ValueError(f"Definition '{def_name}' not found in $defs")
            return definitions[def_name].copy()

        def resolve_refs_recursive(obj: Any, definitions: dict[str, Any]) -> Any:
            """Recursively resolve all $ref references in the schema."""
            if isinstance(obj, dict):
                # If this dict has a $ref, replace it with the actual definition
                if "$ref" in obj:
                    ref_def = resolve_ref(obj["$ref"], definitions)
                    # Merge any additional properties from the current object (except $ref)
                    merged = {**ref_def, **{k: v for k, v in obj.items() if k != "$ref"}}
                    # Recursively resolve any refs in the merged definition
                    return resolve_refs_recursive(merged, definitions)
                else:
                    # Process all values recursively
                    return {k: resolve_refs_recursive(v, definitions) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs_recursive(item, definitions) for item in obj]
            else:
                return obj

        # Resolve all references
        normalized_schema = resolve_refs_recursive(schema_dict, defs)

        # Remove $defs section as it's no longer needed
        if "$defs" in normalized_schema:
            normalized_schema.pop("$defs")

        # Convert additionalProperties to regular properties for Gemini GenAI API
        if for_genai_api:

            def convert_additional_properties_to_properties(schema: dict) -> dict:
                """Recursively convert additionalProperties to regular properties.

                For objects with only additionalProperties (like dict[str, T]),
                we convert the additionalProperties value into a regular property
                to satisfy Gemini's requirement that objects must have non-empty properties.
                """
                if isinstance(schema, dict):
                    # Process nested schemas first
                    if "properties" in schema:
                        for prop_schema in schema["properties"].values():
                            convert_additional_properties_to_properties(prop_schema)
                    if "items" in schema:
                        convert_additional_properties_to_properties(schema["items"])
                    if "anyOf" in schema:
                        for any_of_schema in schema["anyOf"]:
                            convert_additional_properties_to_properties(any_of_schema)
                    if "oneOf" in schema:
                        for one_of_schema in schema["oneOf"]:
                            convert_additional_properties_to_properties(one_of_schema)
                    if "allOf" in schema:
                        for all_of_schema in schema["allOf"]:
                            convert_additional_properties_to_properties(all_of_schema)

                    # Convert additionalProperties to a regular property if object has no properties
                    if (
                        schema.get("type") == "object"
                        and "additionalProperties" in schema
                        and not schema.get("properties")
                    ):
                        additional_props_value = schema["additionalProperties"]
                        # Recursively process the value schema
                        if isinstance(additional_props_value, dict):
                            processed_value = convert_additional_properties_to_properties(additional_props_value)
                            # Convert to a regular property (preserving the type information)
                            schema["properties"] = {"value": processed_value}
                            # Remove additionalProperties since we've converted it
                            schema.pop("additionalProperties", None)
                    else:
                        # Remove additionalProperties if object already has properties
                        schema.pop("additionalProperties", None)

                return schema

            normalized_schema = convert_additional_properties_to_properties(normalized_schema)

        return normalized_schema