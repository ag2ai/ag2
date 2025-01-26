# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import importlib
import pkgutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest


@contextmanager
def add_to_sys_path(path: Optional[Path]) -> Iterator[None]:
    if path is None:
        yield
        return

    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    sys.path.append(str(path))
    try:
        yield
    finally:
        sys.path.remove(str(path))


def list_submodules(module_name: str, *, include_path: Optional[Path] = None, include_root: bool = True) -> list[str]:
    """List all submodules of a given module.

    Args:
        module_name (str): The name of the module to list submodules for.
        include_path (Optional[Path], optional): The path to the module. Defaults to None.
        include_root (bool, optional): Whether to include the root module in the list. Defaults to True.

    Returns:
        list: A list of submodule names.
    """
    with add_to_sys_path(include_path):
        try:
            module = importlib.import_module(module_name)  # nosemgrep
        except Exception:
            return []

        # Get the path of the module. This is necessary to find its submodules.
        module_path = module.__path__

        # Initialize an empty list to store the names of submodules
        submodules = [module_name] if include_root else []

        # Iterate over the submodules in the module's path
        for _, name, ispkg in pkgutil.iter_modules(module_path, prefix=f"{module_name}."):
            # Add the name of each submodule to the list
            submodules.append(name)

            if ispkg:
                submodules.extend(list_submodules(name, include_root=False))

        # Return the list of submodule names
        return submodules


def test_list_submodules() -> None:
    # Specify the name of the module you want to inspect
    module_name = "autogen"

    # Get the list of submodules for the specified module
    submodules = list_submodules(module_name)

    assert len(submodules) > 0
    assert "autogen" in submodules
    assert "autogen.io" in submodules
    assert "autogen.coding.jupyter" in submodules


# todo: we should always run this
@pytest.mark.parametrize("module", list_submodules("autogen"))
def test_submodules(module: str) -> None:
    importlib.import_module(module)  # nosemgrep
