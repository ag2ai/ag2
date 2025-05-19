# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block, skip_on_missing_imports
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
        assert LocalLatLng.model_json_schema() == LatLng.model_json_schema()

    def test_FunctionCallingConfig(self) -> None:  # noqa: N802
        assert LocalFunctionCallingConfig.model_json_schema() == FunctionCallingConfig.model_json_schema()

    def test_RetrievalConfig(self) -> None:  # noqa: N802
        assert LocalRetrievalConfig.model_json_schema() == RetrievalConfig.model_json_schema()

    def test_ToolConfig(self) -> None:  # noqa: N802
        assert LocalToolConfig.model_json_schema() == ToolConfig.model_json_schema()

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
        assert actual != "test"

        actual = LocalTestEnum("TEST_2")

        assert actual == TestEnum("TEST_2")  # type: ignore[comparison-overlap]
        assert actual == TestEnum("test_2")  # type: ignore[comparison-overlap]
        assert LocalTestEnum("test_2") == TestEnum("TEST_2")  # type: ignore[comparison-overlap]
        assert actual.value == "TEST_2"
        assert actual == "TEST_2"
        assert actual != "test_2"

    def test_CommonBaseModel(self) -> None:  # noqa: N802
        assert LocalCommonBaseModel.model_config == CommonBaseModel.model_config
        assert LocalCommonBaseModel.__annotations__ == CommonBaseModel.__annotations__

        local_model_json_schema = LocalCommonBaseModel.model_json_schema()
        google_model_json_schema = CommonBaseModel.model_json_schema()

        # In local model, the title is CommonBaseModel, but in google model, it is BaseModel
        local_model_json_schema.pop("title")
        google_model_json_schema.pop("title")
        assert local_model_json_schema == google_model_json_schema
