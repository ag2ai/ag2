# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.import_utils import optional_import_block, skip_on_missing_imports
from autogen.oai.gemini_types import FunctionCallingConfig as LocalFunctionCallingConfig
from autogen.oai.gemini_types import FunctionCallingConfigDict as LocalFunctionCallingConfigDict
from autogen.oai.gemini_types import FunctionCallingConfigMode as LocalFunctionCallingConfigMode
from autogen.oai.gemini_types import ToolConfig as LocalToolConfig
from autogen.oai.gemini_types import ToolConfigDict as LocalToolConfigDict

with optional_import_block():
    from google.genai.types import (
        FunctionCallingConfig,
        FunctionCallingConfigDict,
        FunctionCallingConfigMode,
        ToolConfig,
        ToolConfigDict,
    )


@skip_on_missing_imports(["google.genai.types"], "gemini")
class TestGeminiTypes:
    def test_FunctionCallingConfigMode(self) -> None:  # noqa: N802
        for v in ["MODE_UNSPECIFIED", "AUTO", "ANY", "NONE"]:
            assert getattr(LocalFunctionCallingConfigMode, v) == getattr(FunctionCallingConfigMode, v)

    def test_FunctionCallingConfig(self) -> None:  # noqa: N802
        assert LocalFunctionCallingConfig.model_json_schema() == FunctionCallingConfig.model_json_schema()

    def test_FunctionCallingConfigDict(self) -> None:  # noqa: N802
        assert LocalFunctionCallingConfigDict.__annotations__.keys() == FunctionCallingConfigDict.__annotations__.keys()

    def test_ToolConfig(self) -> None:  # noqa: N802
        assert LocalToolConfig.model_json_schema() == ToolConfig.model_json_schema()

    def test_ToolConfigDict(self) -> None:  # noqa: N802
        assert LocalToolConfigDict.__annotations__.keys() == ToolConfigDict.__annotations__.keys()
