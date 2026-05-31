# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from autogen.import_utils import optional_import_block, skip_on_missing_imports


def _remove_descriptions(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove description fields from schema recursively.

    Description texts may vary between SDK versions but don't affect functionality.
    """
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "description":
            continue
        elif isinstance(value, dict):
            result[key] = _remove_descriptions(value)
        elif isinstance(value, list):
            result[key] = [_remove_descriptions(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result


def _check_schema_subset(google_schema: dict[str, Any], local_schema: dict[str, Any], path: str = "") -> list[str]:
    """Check that all fields in google_schema exist in local_schema.

    Returns list of missing/mismatched fields. Empty list means compatible.
    Local can have additional fields for forward compatibility with newer SDK versions.
    """
    errors = []

    for key, google_value in google_schema.items():
        current_path = f"{path}.{key}" if path else key

        if key not in local_schema:
            errors.append(f"Missing field: {current_path}")
            continue

        local_value = local_schema[key]

        if isinstance(google_value, dict) and isinstance(local_value, dict):
            # Recursively check nested dicts
            errors.extend(_check_schema_subset(google_value, local_value, current_path))
        elif isinstance(google_value, list) and isinstance(local_value, list):
            # For lists, check that types are compatible (same length and compatible items)
            if len(google_value) != len(local_value):
                # Lists can have different lengths if local has more options
                # Just check that all Google items exist in local
                for i, g_item in enumerate(google_value):
                    if isinstance(g_item, dict):
                        # Find matching dict in local list
                        found = False
                        for l_item in local_value:
                            if isinstance(l_item, dict):
                                sub_errors = _check_schema_subset(g_item, l_item, f"{current_path}[{i}]")
                                if not sub_errors:
                                    found = True
                                    break
                        if not found:
                            errors.append(f"No matching item for: {current_path}[{i}]")
                    elif g_item not in local_value:
                        errors.append(f"Missing list item: {current_path}[{i}]={g_item}")
            else:
                for i, (g_item, l_item) in enumerate(zip(google_value, local_value)):
                    if isinstance(g_item, dict) and isinstance(l_item, dict):
                        errors.extend(_check_schema_subset(g_item, l_item, f"{current_path}[{i}]"))
                    elif g_item != l_item and not (isinstance(g_item, dict) or isinstance(l_item, dict)):
                        errors.append(f"Mismatch at {current_path}[{i}]: {g_item} != {l_item}")
        elif google_value != local_value:
            errors.append(f"Mismatch at {current_path}: {google_value} != {local_value}")

    return errors


from autogen.oai.gemini_types import CaseInSensitiveEnum as LocalCaseInSensitiveEnum
from autogen.oai.gemini_types import CommonBaseModel as LocalCommonBaseModel
from autogen.oai.gemini_types import FunctionCallingConfig as LocalFunctionCallingConfig
from autogen.oai.gemini_types import FunctionCallingConfigMode as LocalFunctionCallingConfigMode
from autogen.oai.gemini_types import LatLng as LocalLatLng
from autogen.oai.gemini_types import RetrievalConfig as LocalRetrievalConfig
from autogen.oai.gemini_types import ToolConfig as LocalToolConfig

with optional_import_block():
    from google.genai._common import BaseModel as CommonBaseModel
    from google.genai._common import CaseInSensitiveEnum
    from google.genai.types import (
        FunctionCallingConfig,
        FunctionCallingConfigMode,
        LatLng,
        RetrievalConfig,
        ToolConfig,
    )


@skip_on_missing_imports(["google.genai.types"], "gemini")
class TestGeminiTypes:
    def test_FunctionCallingConfigMode(self) -> None:  # noqa: N802
        for v in ["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE"]:
            assert getattr(LocalFunctionCallingConfigMode, v) == getattr(FunctionCallingConfigMode, v)

    def test_LatLng(self) -> None:  # noqa: N802
        # Check local schema is compatible with Google SDK schema
        # (local can have extra fields for forward compatibility with newer SDK versions)
        local_schema = _remove_descriptions(LocalLatLng.model_json_schema())
        google_schema = _remove_descriptions(LatLng.model_json_schema())
        errors = _check_schema_subset(google_schema, local_schema)
        assert not errors, f"Schema incompatibility: {errors}"

    def test_FunctionCallingConfig(self) -> None:  # noqa: N802
        # Check local schema is compatible with Google SDK schema
        # (local can have extra fields for forward compatibility with newer SDK versions)
        local_schema = _remove_descriptions(LocalFunctionCallingConfig.model_json_schema())
        google_schema = _remove_descriptions(FunctionCallingConfig.model_json_schema())
        errors = _check_schema_subset(google_schema, local_schema)
        assert not errors, f"Schema incompatibility: {errors}"

    def test_RetrievalConfig(self) -> None:  # noqa: N802
        # Check local schema is compatible with Google SDK schema
        # (local can have extra fields for forward compatibility with newer SDK versions)
        local_schema = _remove_descriptions(LocalRetrievalConfig.model_json_schema())
        google_schema = _remove_descriptions(RetrievalConfig.model_json_schema())
        errors = _check_schema_subset(google_schema, local_schema)
        assert not errors, f"Schema incompatibility: {errors}"

    def test_ToolConfig(self) -> None:  # noqa: N802
        # Check local schema is compatible with Google SDK schema
        # (local can have extra fields for forward compatibility with newer SDK versions)
        local_schema = _remove_descriptions(LocalToolConfig.model_json_schema())
        google_schema = _remove_descriptions(ToolConfig.model_json_schema())
        errors = _check_schema_subset(google_schema, local_schema)
        assert not errors, f"Schema incompatibility: {errors}"

    def test_CaseInSensitiveEnum(self) -> None:  # noqa: N802
        class LocalTestEnum(LocalCaseInSensitiveEnum):
            """Test enum."""

            TEST = "TEST"
            TEST_2 = "TEST_2"

        class TestEnum(CaseInSensitiveEnum):  # type: ignore[misc, no-any-unimported]
            """Test enum."""

            TEST = "TEST"
            TEST_2 = "TEST_2"

        actual = LocalTestEnum("test")

        assert actual == TestEnum("test")  # type: ignore[comparison-overlap]
        assert actual == TestEnum("TEST")  # type: ignore[comparison-overlap]
        assert LocalTestEnum("TEST") == TestEnum("test")  # type: ignore[comparison-overlap]
        assert actual.value == "TEST"
        assert actual == "TEST"
        assert actual != "test"  # type: ignore[comparison-overlap]

        actual = LocalTestEnum("TEST_2")

        assert actual == TestEnum("TEST_2")  # type: ignore[comparison-overlap]
        assert actual == TestEnum("test_2")  # type: ignore[comparison-overlap]
        assert LocalTestEnum("test_2") == TestEnum("TEST_2")  # type: ignore[comparison-overlap]
        assert actual.value == "TEST_2"
        assert actual == "TEST_2"
        assert actual != "test_2"  # type: ignore[comparison-overlap]

    def test_CommonBaseModel(self) -> None:  # noqa: N802
        assert LocalCommonBaseModel.model_config == CommonBaseModel.model_config
        assert LocalCommonBaseModel.__annotations__ == CommonBaseModel.__annotations__

        local_model_json_schema = LocalCommonBaseModel.model_json_schema()
        google_model_json_schema = CommonBaseModel.model_json_schema()

        # In local model, the title is CommonBaseModel, but in google model, it is BaseModel
        local_model_json_schema.pop("title")
        google_model_json_schema.pop("title")
        assert local_model_json_schema == google_model_json_schema
