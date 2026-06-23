# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from autogen.import_utils import run_for_optional_imports
from autogen.interop import Interoperability


@pytest.mark.interop
class TestInteroperability:
    def test_supported_types(self) -> None:
        actual = Interoperability.get_supported_types()

        if sys.version_info >= (3, 9) and sys.version_info < (3, 10):
            assert actual == ["langchain", "pydanticai"]

        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            assert actual == ["langchain", "pydanticai"]

        if sys.version_info >= (3, 13):
            assert actual == ["langchain", "pydanticai"]

    def test_unsupported_type_error_message(self) -> None:
        """The error for an unsupported interop type should list the actual type names."""
        from unittest.mock import MagicMock, patch

        mock_registry = MagicMock()
        mock_registry.get_supported_types.return_value = ["langchain", "pydanticai"]

        with patch.object(Interoperability, "registry", mock_registry), pytest.raises(ValueError, match="'langchain'"):
            Interoperability.get_interoperability_class("nonexistent")

    @pytest.mark.skip("This test is not yet implemented")
    @run_for_optional_imports("langchain", "interop-langchain")
    def test_langchain(self) -> None:
        raise NotImplementedError("This test is not yet implemented")
