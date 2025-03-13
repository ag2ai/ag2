# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
import sys
from types import ModuleType
from typing import Any, Iterable, Iterator, Optional, Type, Union

import pytest

from autogen.import_utils import ModuleInfo, get_missing_imports, optional_import_block, require_optional_import


@pytest.fixture
def mock_module() -> Iterator[ModuleType]:
    module_name = "mock_module"
    module = ModuleType(module_name)
    module.__version__ = "1.0.0"  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    yield module
    del sys.modules[module_name]


@pytest.fixture
def mock_module_without_version() -> Iterator[ModuleType]:
    module_name = "mock_module"
    module = ModuleType(module_name)
    sys.modules[module_name] = module
    yield module
    del sys.modules[module_name]


class MockModule:
    def __init__(self, name: str, version: str):
        self.__name__ = name
        self.__version__ = version


@pytest.fixture
def mock_modules() -> Iterator[dict[str, ModuleType]]:
    modules = {
        "module_a": MockModule("module_a", "1.0.0"),
        "module_b": MockModule("module_b", "2.0.0"),
        "module_c": MockModule("module_c", "3.0.0"),
    }
    original_sys_modules = sys.modules.copy()
    sys.modules.update(modules)  # type: ignore[arg-type]
    assert all(module in sys.modules for module in modules)
    try:
        yield modules  # type: ignore[misc]
    finally:
        sys.modules.clear()
        sys.modules.update(original_sys_modules)


class TestmoduleInfo:
    @pytest.mark.parametrize(
        "module_info, expected",
        [
            (
                "jupyter-client>=8.6.0,<9.0.0",
                ModuleInfo(
                    name="jupyter-client",
                    min_version="8.6.0",
                    max_version="9.0.0",
                    min_inclusive=True,
                    max_inclusive=False,
                ),
            ),
            (
                "jupyter-client>=8.6.0",
                ModuleInfo(
                    name="jupyter-client",
                    min_version="8.6.0",
                    max_version=None,
                    min_inclusive=True,
                    max_inclusive=False,
                ),
            ),
            (
                "jupyter-client<9.0.0",
                ModuleInfo(
                    name="jupyter-client",
                    min_version=None,
                    max_version="9.0.0",
                    min_inclusive=False,
                    max_inclusive=False,
                ),
            ),
            (
                "jupyter-client",
                ModuleInfo(
                    name="jupyter-client", min_version=None, max_version=None, min_inclusive=False, max_inclusive=False
                ),
            ),
        ],
    )
    def test_from_str_success(self, module_info: str, expected: ModuleInfo) -> None:
        result = ModuleInfo.from_str(module_info)
        assert result == expected

    def test_from_str_with_invalid_format(self) -> None:
        module_info = "jupyter-client>="
        with pytest.raises(ValueError, match="Invalid module information: jupyter-client>="):
            ModuleInfo.from_str(module_info)

    @pytest.mark.parametrize(
        "module_info, expected",
        [
            (ModuleInfo(name="mock_module"), None),
            (ModuleInfo(name="non_existent_module"), "'non_existent_module' is not installed."),
            (ModuleInfo(name="mock_module", min_version="1.0.0", min_inclusive=True), None),
            (
                ModuleInfo(name="mock_module", min_version="1.0.0", min_inclusive=False),
                "'mock_module' is installed, but the installed version 1.0.0 is too low (required 'mock_module>1.0.0').",
            ),
            (ModuleInfo(name="mock_module", min_version="0.9.0", min_inclusive=True), None),
            (
                ModuleInfo(name="mock_module", min_version="1.1.0", min_inclusive=True),
                "'mock_module' is installed, but the installed version 1.0.0 is too low (required 'mock_module>=1.1.0').",
            ),
            (ModuleInfo(name="mock_module", max_version="1.0.0", max_inclusive=True), None),
            (
                ModuleInfo(name="mock_module", max_version="1.0.0", max_inclusive=False),
                "'mock_module' is installed, but the installed version 1.0.0 is too high (required 'mock_module<1.0.0').",
            ),
            (ModuleInfo(name="mock_module", max_version="1.1.0", max_inclusive=True), None),
            (
                ModuleInfo(name="mock_module", max_version="0.9.0", max_inclusive=True),
                "'mock_module' is installed, but the installed version 1.0.0 is too high (required 'mock_module<=0.9.0').",
            ),
            (
                ModuleInfo(
                    name="mock_module", min_version="0.9.0", max_version="1.1.0", min_inclusive=True, max_inclusive=True
                ),
                None,
            ),
            (
                ModuleInfo(
                    name="mock_module", min_version="1.0.0", max_version="1.0.0", min_inclusive=True, max_inclusive=True
                ),
                None,
            ),
            (
                ModuleInfo(
                    name="mock_module",
                    min_version="1.0.0",
                    max_version="1.0.0",
                    min_inclusive=False,
                    max_inclusive=False,
                ),
                "'mock_module' is installed, but the installed version 1.0.0 is too low (required 'mock_module>1.0.0<1.0.0').",
            ),
        ],
    )
    def test_is_in_sys_modules(self, mock_module: ModuleType, module_info: ModuleInfo, expected: Optional[str]) -> None:
        assert module_info.is_in_sys_modules() == expected

    @pytest.mark.parametrize(
        "module_info, expected",
        [
            (ModuleInfo(name="mock_module"), None),
            (
                ModuleInfo(name="mock_module", min_version="1.0.0", min_inclusive=True),
                "'mock_module' is installed, but the version is not available.",
            ),
        ],
    )
    def test_is_in_sys_modules_without_version(
        self, mock_module_without_version: ModuleType, module_info: ModuleInfo, expected: Optional[str]
    ) -> None:
        assert module_info.is_in_sys_modules() == expected


class TestOptionalImportBlock:
    def test_optional_import_block(self) -> None:
        with optional_import_block():
            import ast

            import some_module
            import some_other_module

        assert ast is not None
        with pytest.raises(
            UnboundLocalError,
            match=r"(local variable 'some_module' referenced before assignment|cannot access local variable 'some_module' where it is not associated with a value)",
        ):
            some_module

        with pytest.raises(
            UnboundLocalError,
            match=r"(local variable 'some_other_module' referenced before assignment|cannot access local variable 'some_other_module' where it is not associated with a value)",
        ):
            some_other_module


class TestRequiresOptionalImportCallables:
    def test_version_too_high(self, mock_module: ModuleInfo) -> None:
        assert mock_module.__name__ in sys.modules  # type: ignore[attr-defined]
        mock_info = f"{mock_module.__name__}>{mock_module.__version__}"  # type: ignore[attr-defined]

        @require_optional_import(mock_info, "mock-module")
        def dummy_function() -> None:
            """Dummy function to test requires_optional_import"""
            pass

        with pytest.raises(
            ImportError,
            match=re.escape("""A module needed for test.test_import_utils.dummy_function is missing:
 - 'mock_module' is installed, but the installed version 1.0.0 is too low (required 'mock_module>1.0.0').
Please install it using:
'pip install ag2[mock-module]'"""),
        ):
            dummy_function()

    @pytest.mark.parametrize("except_for", [None, "dummy_function", ["dummy_function"]])
    def test_function_attributes(self, except_for: Optional[Union[str, list[str]]]) -> None:
        def dummy_function() -> None:
            """Dummy function to test requires_optional_import"""
            pass

        dummy_function.__module__ = "some_random_module.dummy_stuff"

        actual = require_optional_import("some_optional_module", "optional_dep", except_for=except_for)(dummy_function)

        assert actual is not None
        assert actual.__module__ == "some_random_module.dummy_stuff"
        assert actual.__name__ == "dummy_function"
        assert actual.__doc__ == "Dummy function to test requires_optional_import"

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for some_random_module.dummy_stuff.dummy_function is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]"""),
            ):
                actual()
        else:
            actual()

    @pytest.mark.parametrize("except_for", [None, "dummy_function", ["dummy_function"]])
    def test_function_call(self, except_for: Optional[Union[str, list[str]]]) -> None:
        @require_optional_import("some_optional_module", "optional_dep", except_for=except_for)
        def dummy_function() -> None:
            """Dummy function to test requires_optional_import"""
            pass

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for test.test_import_utils.dummy_function is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]'"""),
            ):
                dummy_function()
        else:
            dummy_function()

    @pytest.mark.parametrize("except_for", [None, "dummy_method", ["dummy_method"]])
    def test_method_attributes(self, except_for: Optional[Union[str, list[str]]]) -> None:
        class DummyClass:
            def dummy_method(self) -> None:
                """Dummy method to test requires_optional_import"""
                pass

        assert hasattr(DummyClass.dummy_method, "__module__")
        assert DummyClass.dummy_method.__module__ == "test.test_import_utils"

        DummyClass.__module__ = "some_random_module.dummy_stuff"
        DummyClass.dummy_method.__module__ = "some_random_module.dummy_stuff"

        DummyClass.dummy_method = require_optional_import(  # type: ignore[method-assign]
            "some_optional_module", "optional_dep", except_for=except_for
        )(DummyClass.dummy_method)

        assert DummyClass.dummy_method is not None
        assert DummyClass.dummy_method.__module__ == "some_random_module.dummy_stuff"
        assert DummyClass.dummy_method.__name__ == "dummy_method"
        assert DummyClass.dummy_method.__doc__ == "Dummy method to test requires_optional_import"

        dummy = DummyClass()

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for some_random_module.dummy_stuff.dummy_method is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]'"""),
            ):
                dummy.dummy_method()
        else:
            dummy.dummy_method()

    @pytest.mark.parametrize("except_for", [None, "dummy_method", ["dummy_method"]])
    def test_method_call(self, except_for: Optional[Union[str, list[str]]]) -> None:
        class DummyClass:
            @require_optional_import("some_optional_module", "optional_dep", except_for=except_for)
            def dummy_method(self) -> None:
                """Dummy method to test requires_optional_import"""
                pass

        dummy = DummyClass()

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for test.test_import_utils.dummy_method is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]'"""),
            ):
                dummy.dummy_method()
        else:
            dummy.dummy_method()

    @pytest.mark.parametrize("except_for", [None, "dummy_static_function", ["dummy_static_function"]])
    def test_static_call(self, except_for: Optional[Union[str, list[str]]]) -> None:
        class DummyClass:
            @require_optional_import("some_optional_module", "optional_dep", except_for=except_for)
            @staticmethod
            def dummy_static_function() -> None:
                """Dummy static function to test requires_optional_import"""
                pass

        dummy = DummyClass()

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for test.test_import_utils.dummy_static_function is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]'"""),
            ):
                dummy.dummy_static_function()
        else:
            dummy.dummy_static_function()

    @pytest.mark.parametrize("except_for", [None, "dummy_property", ["dummy_property"]])
    def test_property_call(self, except_for: Optional[Union[str, list[str]]]) -> None:
        class DummyClass:
            @property
            @require_optional_import("some_optional_module", "optional_dep", except_for=except_for)
            def dummy_property(self) -> int:
                """Dummy property to test requires_optional_import"""
                return 4

        dummy = DummyClass()

        if not except_for:
            with pytest.raises(
                ImportError,
                match=re.escape("""A module needed for test.test_import_utils.dummy_property is missing:
 - 'some_optional_module' is not installed.
Please install it using:
'pip install ag2[optional_dep]'"""),
            ):
                dummy.dummy_property

        else:
            dummy.dummy_property


class TestRequiresOptionalImportClasses:
    @pytest.fixture
    def dummy_cls(self) -> Type[Any]:
        @require_optional_import("some_optional_module", "optional_dep")
        class DummyClass:
            def dummy_method(self) -> None:
                """Dummy method to test requires_optional_import"""
                pass

            @staticmethod
            def dummy_static_method() -> None:
                """Dummy static method to test requires_optional_import"""
                pass

            @classmethod
            def dummy_class_method(cls) -> None:
                """Dummy class method to test requires_optional_import"""
                pass

            @property
            def dummy_property(self) -> int:
                """Dummy property to test requires_optional_import"""
                return 4

        return DummyClass

    def test_class_init_call(self, dummy_cls: Type[Any]) -> None:
        with pytest.raises(
            ImportError,
            match=re.escape("""A module needed for __init__ is missing:
 - 'some_optional_module' is not installed.
Please install it using:\n'pip install ag2[optional_dep]'"""),
        ):
            dummy_cls()


class TestGetMissingImports:
    @pytest.mark.parametrize(
        "modules, expected_missing",
        [
            (["module_a", "module_b", "module_c"], {}),
            (["module_a>=1.0.0", "module_b>=2.0.0", "module_c>=3.0.0"], {}),
            (
                ["module_a>=1.0.1", "module_b>=2.0.0", "module_c>=3.0.0"],
                {
                    "module_a": "'module_a' is installed, but the installed version 1.0.0 is too low (required 'module_a>=1.0.1')."
                },
            ),
            (
                ["module_a>=1.0.0", "module_b>=2.1.0", "module_c>=3.0.0"],
                {
                    "module_b": "'module_b' is installed, but the installed version 2.0.0 is too low (required 'module_b>=2.1.0')."
                },
            ),
            (
                ["module_a>=1.0.0", "module_b>=2.0.0", "module_c>=3.1.0"],
                {
                    "module_c": "'module_c' is installed, but the installed version 3.0.0 is too low (required 'module_c>=3.1.0')."
                },
            ),
            (["module_a>=1.0.0", "module_b>=2.0.0", "module_d"], {"module_d": "'module_d' is not installed."}),
            (
                ["module_a>=1.0.0", "module_b>=2.0.0", "module_c<3.0.0"],
                {
                    "module_c": "'module_c' is installed, but the installed version 3.0.0 is too high (required 'module_c<3.0.0')."
                },
            ),
            (["module_a>=1.0.0", "module_b>=2.0.0", "module_c<=3.0.0"], {}),
            (
                ["module_a>=1.0.0", "module_b>=2.0.0", "module_c>3.0.0"],
                {
                    "module_c": "'module_c' is installed, but the installed version 3.0.0 is too low (required 'module_c>3.0.0')."
                },
            ),
            (["module_a>=1.0.0", "module_b>=2.0.0", "module_c<3.1.0"], {}),
        ],
    )
    def test_get_missing_imports(
        self, mock_modules: dict[str, MockModule], modules: Union[str, Iterable[str]], expected_missing: dict[str, str]
    ) -> None:
        assert mock_modules
        missing = get_missing_imports(modules)
        assert missing == expected_missing
