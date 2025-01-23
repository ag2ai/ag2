#!/usr/bin/env python

# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import importlib
import pkgutil
import shutil
from collections.abc import Iterable
from pathlib import Path

import pdoc


def import_submodules(module_name: str, *, include_root: bool = True) -> list[str]:
    """List all submodules of a given module.

    Args:
        module_name (str): The name of the module to list submodules for.
        include_path (Optional[Path], optional): The path to the module. Defaults to None.
        include_root (bool, optional): Whether to include the root module in the list. Defaults to True.

    Returns:
        list: A list of submodule names.
    """
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
            submodules.extend(import_submodules(name, include_root=False))

    # Return the list of submodule names
    return submodules


def build_pdoc_dict(module_name: str) -> None:
    module = importlib.import_module(module_name)  # nosemgrep

    if not hasattr(module, "__pdoc__"):
        setattr(module, "__pdoc__", {})

    for name, obj in module.__dict__.items():
        if not hasattr(obj, "__name__") or name.startswith("_"):
            continue

        if hasattr(obj, "__module__") and obj.__module__ != module_name:
            print(f"Skipping {obj.__module__}.{obj.__name__} because it is not from {module_name}")
            module.__pdoc__[name] = False


def generate_markdown(path: Path) -> None:
    modules = ["autogen"]  # Public submodules are auto-imported
    context = pdoc.Context()

    modules = [pdoc.Module(mod, context=context) for mod in modules]
    pdoc.link_inheritance(context)

    def recursive_markdown(mod: pdoc.Module) -> Iterable[tuple[str, str]]:  # type: ignore[no-any-unimported]
        yield mod.name, mod.text()
        for submod in mod.submodules():
            yield from recursive_markdown(submod)

    for mod in modules:
        # todo: pass the template here
        for module_name, text in recursive_markdown(mod):
            file_path = path / module_name.replace(".", "/") / "index.md"
            # print(f"Writing {file_path}...")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w") as f:
                f.write(text)


submodules = import_submodules("autogen")
print(f"{submodules=}")

for submodule in submodules:
    build_pdoc_dict(submodule)

shutil.rmtree("tmp", ignore_errors=True)
generate_markdown(Path("tmp"))
