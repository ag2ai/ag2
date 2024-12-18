# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#

import importlib
import inspect
import logging
import pkgutil
import sys
from typing import Any, Dict, List, Set, Type

from .interoperable import Interoperable

logger = logging.getLogger(__name__)


def import_submodules(package_name: str) -> List[str]:
    package = importlib.import_module(package_name)
    imported_modules: List[str] = []
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(module_name)

            imported_modules.append(module_name)
        except Exception as e:
            logger.info(f"Error importing {module_name}, most likely perfectly fine: {e}")

    return imported_modules


def find_classes_implementing_protocol(imported_modules: List[str], protocol: Type[Any]) -> List[Type[Any]]:
    implementing_classes: Set[Type[Any]] = set()
    for module in imported_modules:
        for _, obj in inspect.getmembers(sys.modules[module], inspect.isclass):
            if issubclass(obj, protocol) and obj is not protocol:
                implementing_classes.add(obj)

    return list(implementing_classes)


def get_all_interoperability_classes() -> Dict[str, Type[Interoperable]]:
    imported_modules = import_submodules("autogen.interop")
    classes = find_classes_implementing_protocol(imported_modules, Interoperable)

    # check that all classes finish with 'Interoperability'
    for cls in classes:
        if not cls.__name__.endswith("Interoperability"):
            raise RuntimeError(f"Class {cls} does not end with 'Interoperability'")

    retval = {
        cls.__name__.split("Interoperability")[0].lower(): cls for cls in classes if cls.__name__ != "Interoperability"
    }

    return retval
