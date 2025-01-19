# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Type

import pytest

from autogen.import_utils import optional_import_block, require_optional_import


class TestOptionalImportBlock:
    def test_optional_import_block(self):
        with optional_import_block():
            import ast

            import some_module
            import some_other_module

        assert ast is not None
        with pytest.raises(UnboundLocalError, match="local variable 'some_module' referenced before assignment"):
            some_module

        with pytest.raises(UnboundLocalError, match="local variable 'some_other_module' referenced before assignment"):
            some_other_module


class TestRequiresOptionalImportCallables:
    def test_function_attributes(self):
        def dummy_function():
            """Dummy function to test requires_optional_import"""
            pass

        dummy_function.__module__ = "some_random_module.dummy_stuff"

        actual = require_optional_import("some_optional_module", "optional_dep")(dummy_function)

        assert actual is not None
        assert actual.__module__ == "some_random_module.dummy_stuff"
        assert actual.__name__ == "dummy_function"
        assert actual.__doc__ == "Dummy function to test requires_optional_import"

        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for some_random_module.dummy_stuff.dummy_function is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            actual()

    def test_function_call(self):
        @require_optional_import("some_optional_module", "optional_dep")
        def dummy_function():
            """Dummy function to test requires_optional_import"""
            pass

        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for test.test_import_utils.dummy_function is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            dummy_function()

    def test_method_attributes(self):
        class DummyClass:
            def dummy_method(self):
                """Dummy method to test requires_optional_import"""
                pass

        assert hasattr(DummyClass.dummy_method, "__module__")
        assert DummyClass.dummy_method.__module__ == "test.test_import_utils"

        DummyClass.__module__ = "some_random_module.dummy_stuff"
        DummyClass.dummy_method.__module__ = "some_random_module.dummy_stuff"

        DummyClass.dummy_method = require_optional_import("some_optional_module", "optional_dep")(
            DummyClass.dummy_method
        )

        assert DummyClass.dummy_method is not None
        assert DummyClass.dummy_method.__module__ == "some_random_module.dummy_stuff"
        assert DummyClass.dummy_method.__name__ == "dummy_method"
        assert DummyClass.dummy_method.__doc__ == "Dummy method to test requires_optional_import"

        dummy = DummyClass()
        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for some_random_module.dummy_stuff.dummy_method is missing, please install it using 'pip install ag2\[optional_dep\]",
        ):
            dummy.dummy_method()

    def test_method_call(self):
        class DummyClass:
            @require_optional_import("some_optional_module", "optional_dep")
            def dummy_method(self):
                """Dummy method to test requires_optional_import"""
                pass

        dummy = DummyClass()
        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for test.test_import_utils.dummy_method is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            dummy.dummy_method()

    def test_static_call(self):
        class DummyClass:
            @require_optional_import("some_optional_module", "optional_dep")
            @staticmethod
            def dummy_static_function():
                """Dummy static function to test requires_optional_import"""
                pass

        dummy = DummyClass()
        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for test.test_import_utils.dummy_static_function is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            dummy.dummy_static_function()

    def test_property_call(self):
        class DummyClass:
            @require_optional_import("some_optional_module", "optional_dep")
            @property
            def dummy_property(self) -> int:
                """Dummy property to test requires_optional_import"""
                return 4

        dummy = DummyClass()
        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for test.test_import_utils.dummy_property is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            dummy.dummy_property


class TestRequiresOptionalImportClasses:
    @pytest.fixture
    def dummy_cls(self) -> Type[Any]:
        @require_optional_import("some_optional_module", "optional_dep")
        class DummyClass:
            def dummy_method(self):
                """Dummy method to test requires_optional_import"""
                pass

            @staticmethod
            def dummy_static_method():
                """Dummy static method to test requires_optional_import"""
                pass

            @classmethod
            def dummy_class_method(cls):
                """Dummy class method to test requires_optional_import"""
                pass

            @property
            def dummy_property(self) -> int:
                """Dummy property to test requires_optional_import"""
                return 4

        return DummyClass

    def test_class_init_call(self, dummy_cls: Type[Any]):
        with pytest.raises(
            ImportError,
            match=r"Module 'some_optional_module' needed for __init__ is missing, please install it using 'pip install ag2\[optional_dep\]'",
        ):
            dummy_cls()
