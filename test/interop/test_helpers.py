# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

from autogen.interop import Interoperability, Interoperable
from autogen.interop.helpers import (
    find_classes_implementing_protocol,
    get_all_interoperability_classes,
    import_submodules,
)


class TestHelpers:
    @pytest.fixture(autouse=True)
    def setup_method(self) -> None:
        self.imported_modules = import_submodules("autogen.interop")

    def test_import_submodules(self) -> None:
        assert "autogen.interop.helpers" in self.imported_modules

    def test_find_classes_implementing_protocol(self) -> None:
        actual = find_classes_implementing_protocol(self.imported_modules, Interoperable)
        print(f"test_find_classes_implementing_protocol: {actual=}")

        assert Interoperability in actual
        expected_count = 1

        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            from autogen.interop.crewai import CrewAIInteroperability

            assert CrewAIInteroperability in actual
            expected_count += 1

        if sys.version_info >= (3, 9):
            from autogen.interop.langchain import LangchainInteroperability

            assert LangchainInteroperability in actual
            expected_count += 1

        assert len(actual) == expected_count

    def test_get_all_interoperability_classes(self) -> None:

        actual = get_all_interoperability_classes()

        if sys.version_info < (3, 9):
            assert actual == {}

        if sys.version_info >= (3, 10) and sys.version_info < (3, 13):
            from autogen.interop.crewai import CrewAIInteroperability
            from autogen.interop.langchain import LangchainInteroperability

            assert actual == {"crewai": CrewAIInteroperability, "langchain": LangchainInteroperability}

        if (sys.version_info >= (3, 9) and sys.version_info < (3, 10)) and sys.version_info >= (3, 13):
            from autogen.interop.langchain import LangchainInteroperability

            assert actual == {"langchain": LangchainInteroperability}
