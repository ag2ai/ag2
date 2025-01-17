# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.exception_utils import AutogenImportError
from autogen.handle_imports import check_for_missing_imports, requires_optional_import


def test_check_for_missing_imports():
    with check_for_missing_imports():
        pass  # Safe to attempt, even if it fails
    assert True


class TestRequiresOptionalImport:
    def test_with_class(
        self,
    ):
        @requires_optional_import("some_optional_module", "test")
        class DummyClass:
            def __init__(self):
                o = some_optional_module.SomeClass()  # Raises MissingImportError if module is missing

        assert DummyClass is not None
        with pytest.raises(AutogenImportError) as e:
            DummyClass()

        assert (
            str(e.value)
            == "Missing imported module 'some_optional_module', please install it using 'pip install ag2[test]'"
        )

    def test_with_function(self):
        @requires_optional_import("some_other_optional_module", "test")
        def dummy_function():
            o = some_other_optional_module.SomeOtherClass()

        assert dummy_function is not None
        with pytest.raises(AutogenImportError) as e:
            dummy_function()

        assert (
            str(e.value)
            == "Missing imported module 'some_other_optional_module', please install it using 'pip install ag2[test]'"
        )

    def test_with_multiple_modules(self):
        @requires_optional_import(["module1", "module2"], "test")
        def dummy_function():
            o = module1.SomeClass()
            o2 = module2.SomeOtherClass()

        assert dummy_function is not None
        with pytest.raises(AutogenImportError) as e:
            dummy_function()

        assert (
            str(e.value)
            == "Missing imported module 'module1', 'module2', please install it using 'pip install ag2[test]'"
        )
