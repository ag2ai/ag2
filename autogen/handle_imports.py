# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import importlib
from contextlib import contextmanager
from functools import wraps
from typing import Generator, Iterable, Union

from autogen.exception_utils import AutogenImportError


@contextmanager
def check_for_missing_imports() -> Generator[None, None, None]:
    """
    A context manager to temporarily suppress ImportErrors.
    Use this to attempt imports without failing immediately on missing modules.
    """
    try:
        yield
    except ImportError:
        pass  # Ignore ImportErrors during this context


# class DummyModule:
#     """A dummy module that raises ImportError when any attribute is accessed"""

#     def __init__(self, name: str, dep_target: str):
#         self._name = name
#         self._dep_target = dep_target

#     def __getattr__(self, attr: str) -> Any:
#         raise AutogenImportError(missing_modules=self._name, dep_target=self._dep_target)


def requires_optional_import(modules: Union[str, Iterable[str]], dep_target: str):
    """Decorator to handle optional module dependencies

    Args:
        modules: Module name or list of module names required
        dep_target: Target name for pip installation (e.g. 'test' in pip install ag2[test])
    """
    if isinstance(modules, str):
        modules = [modules]

    def decorator(cls):
        # Check if all required modules are available
        missing_modules = []
        dummy_modules = {}

        for module_name in modules:
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing_modules.append(module_name)
                # Create dummy module
                # dummy_module = DummyModule(module_name, dep_target)
                # dummy_modules[module_name] = dummy_module
                # sys.modules[module_name] = dummy_module

        if missing_modules:
            # Replace real class with dummy that raises ImportError
            @wraps(cls)
            def dummy_class(*args, **kwargs):
                raise AutogenImportError(missing_modules=missing_modules, dep_target=dep_target)

            return dummy_class
        return cls

    return decorator
