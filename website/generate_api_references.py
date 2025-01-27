# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
import os
import pkgutil
import shutil
import sys
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import pdoc
from jinja2 import Template


@contextmanager
def windows_encoding() -> Iterator[str]:
    """Context manager to handle Windows-specific encoding.

    Yields:
        str: The encoding to use for file operations ('utf-16' on Windows, 'utf-8' otherwise)
    """
    encoding = "utf-16" if sys.platform == "win32" else "utf-8"
    original_value = os.environ.get("PYTHONIOENCODING") if sys.platform == "win32" else None

    if sys.platform == "win32":
        os.environ["PYTHONIOENCODING"] = "utf16"

    try:
        yield encoding
    finally:
        if sys.platform == "win32":
            if original_value is not None:
                os.environ["PYTHONIOENCODING"] = original_value
            else:
                os.environ.pop("PYTHONIOENCODING", None)


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


def generate_markdown(path: Path, encoding: str) -> None:
    modules = ["autogen"]  # Public submodules are auto-imported
    context = pdoc.Context()

    modules = [pdoc.Module(mod, context=context) for mod in modules]
    pdoc.link_inheritance(context)

    def recursive_markdown(mod: pdoc.Module) -> Iterable[tuple[str, str]]:  # type: ignore[no-any-unimported]
        # Pass our custom template here
        yield mod.name, mod.text()
        for submod in mod.submodules():
            yield from recursive_markdown(submod)

    for mod in modules:
        for module_name, text in recursive_markdown(mod):
            file_path = path / module_name.replace(".", "/") / "index.md"
            # print(f"Writing {file_path}...")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with file_path.open("w", encoding=encoding) as f:
                f.write(text)


def generate(target_dir: Path, template_dir: Path, encoding: str) -> None:
    # Pass the custom template directory for rendering the markdown
    pdoc.tpl_lookup.directories.insert(0, str(template_dir))

    submodules = import_submodules("autogen")
    print(f"{submodules=}")

    for submodule in submodules:
        build_pdoc_dict(submodule)

    generate_markdown(target_dir, encoding)


def read_file_content(file_path: Path, encoding: str) -> str:
    """Read content from a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: Content of the file
    """
    with open(file_path, encoding=encoding) as f:
        return f.read()


def write_file_content(file_path: str, content: str) -> None:
    """Write content to a file.

    Args:
        file_path (str): Path to the file
        content (str): Content to write
    """
    encoding = "utf-16" if sys.platform == "win32" else "utf-8"
    with open(file_path, "w", encoding=encoding) as f:
        f.write(content)


def convert_md_to_mdx(input_dir: Path, encoding: str) -> None:
    """Convert all .md files in directory to .mdx while preserving structure.

    Args:
        input_dir (Path): Directory containing .md files to convert
    """
    if not input_dir.exists():
        print(f"Directory not found: {input_dir}")
        sys.exit(1)

    for md_file in input_dir.rglob("*.md"):
        mdx_file = md_file.with_suffix(".mdx")

        # Read content from .md file
        content = md_file.read_text(encoding=encoding)

        # Write content to .mdx file
        mdx_file.write_text(content, encoding=encoding)

        # Remove original .md file
        md_file.unlink()
        print(f"Converted: {md_file} -> {mdx_file}")


def get_mdx_files(directory: Path) -> list[str]:
    """Get all MDX files in directory and subdirectories."""
    return [f"{p.relative_to(directory).with_suffix('')!s}".replace("\\", "/") for p in directory.rglob("*.mdx")]


def add_prefix(path: str, parent_groups: Optional[list[str]] = None) -> str:
    """Create full path with prefix and parent groups."""
    groups = parent_groups or []
    return f"reference/{'/'.join(groups + [path])}"


def create_nav_structure(paths: list[str], parent_groups: Optional[list[str]] = None) -> list[Any]:
    """Convert list of file paths into nested navigation structure."""
    groups: dict[str, list[str]] = {}
    pages = []
    parent_groups = parent_groups or []

    for path in paths:
        parts = path.split("/")
        if len(parts) == 1:
            pages.append(add_prefix(path, parent_groups))
        else:
            group = parts[0]
            subpath = "/".join(parts[1:])
            groups.setdefault(group, []).append(subpath)

    # Sort directories and create their structures
    sorted_groups = [
        {
            "group": group,
            "pages": create_nav_structure(subpaths, parent_groups + [group]),
        }
        for group, subpaths in sorted(groups.items())
    ]

    # Sort pages
    overview_page = [page for page in pages if page.endswith("overview")]
    if overview_page:
        pages.remove(overview_page[0])

    sorted_pages = sorted(pages)
    if overview_page:
        sorted_pages.insert(0, overview_page[0])

    # Return directories first, then files
    return sorted_pages + sorted_groups


def update_nav(mint_json_path: Path, new_nav_pages: list[Any], encoding: str) -> None:
    """Update the 'API Reference' section in mint.json navigation with new pages.

    Args:
        mint_json_path: Path to mint.json file
        new_nav_pages: New navigation structure to replace in API Reference pages
    """
    try:
        # Read the current mint.json
        with open(mint_json_path) as f:
            mint_config = json.load(f)

        reference_section = {"group": "API Reference", "pages": new_nav_pages}
        mint_config["navigation"].append(reference_section)

        # Write back to mint.json with proper formatting
        with open(mint_json_path, "w", encoding=encoding) as f:
            json.dump(mint_config, f, indent=2)
            f.write("\n")

    except json.JSONDecodeError:
        print(f"Error: {mint_json_path} is not valid JSON")
    except Exception as e:
        print(f"Error updating mint.json: {e}")


def update_mint_json_with_api_nav(script_dir: Path, api_dir: Path, encoding: str) -> None:
    """Update mint.json with MDX files in the API directory."""
    mint_json_path = script_dir / "mint.json"
    if not mint_json_path.exists():
        print(f"File not found: {mint_json_path}")
        sys.exit(1)

    # Get all MDX files in the API directory
    mdx_files = get_mdx_files(api_dir)

    # Create navigation structure
    nav_structure = create_nav_structure(mdx_files)

    # Update mint.json with new navigation
    update_nav(mint_json_path, nav_structure, encoding)


def generate_mint_json_from_template(mint_json_template_path: Path, mint_json_path: Path, encoding: str) -> None:
    # if mint.json already exists, delete it
    if mint_json_path.exists():
        os.remove(mint_json_path)

    # Copy the template file to mint.json
    contents = read_file_content(mint_json_template_path, encoding)
    mint_json_template_content = Template(contents).render()

    # Parse the rendered template content as JSON
    mint_json_data = json.loads(mint_json_template_content)

    # Write content to mint.json
    with open(mint_json_path, "w", encoding=encoding) as f:
        json.dump(mint_json_data, f, indent=2)


class SplitReferenceFilesBySymbols:
    def __init__(self, api_dir: Path) -> None:
        self.api_dir = api_dir
        self.tmp_dir = Path("tmp")

    def _generate_overview(self, classes: list[str], functions: list[str], output_dir: Path) -> str:
        overview = """---
sidebarTitle: Overview
title: Overview
---
"""
        if classes:
            overview += "\n\n## Classes\n"
            for symbol in sorted(classes):
                href = output_dir / symbol
                overview += f"""<p class="overview-symbol"><a href="/{str(href).replace("tmp/", "reference/")}"><code>class {symbol}</code></a></p>"""
            overview += "\n"

        if functions:
            overview += "\n\n## Functions\n"
            for symbol in sorted(functions):
                href = output_dir / symbol
                overview += f"""<p class="overview-symbol"><a href="/{str(href).replace("tmp/", "reference/")}"><code>{symbol}</code></a></p>"""
            overview += "\n"

        return overview

    def _extract_symbol_content(self, content: str, output_dir: Path) -> dict[str, str]:
        sections = {}
        class_symbols = []
        function_symbols = []

        for part in content.split("**** SYMBOL_START ****")[1:]:
            symbol = part.split("```python\n")[1].split("(")[0].strip()
            content = part.split("**** SYMBOL_END ****")[0].strip()
            sections[symbol] = content

            if "doc-symbol-class" in content:
                class_symbols.append(symbol)

            if "doc-symbol-function" in content:
                function_symbols.append(symbol)

        sections["overview"] = self._generate_overview(class_symbols, function_symbols, output_dir)
        return sections

    def _split_content_by_symbols(self, content: str, output_dir: Path) -> dict[str, str]:
        symbols = {}
        if "**** SYMBOL_START ****" in content:
            symbols.update(self._extract_symbol_content(content, output_dir))
        return symbols

    def _process_files(self) -> Iterator[tuple[Path, dict[str, str]]]:
        for md_file in self.api_dir.rglob("*.md"):
            output_dir = self.tmp_dir / md_file.relative_to(self.api_dir).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            encoding = "utf-16" if sys.platform == "win32" else "utf-8"
            yield output_dir, self._split_content_by_symbols(md_file.read_text(encoding=encoding), output_dir)

    def _clean_directory(self, directory: Path) -> None:
        for item in directory.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def _move_generated_files_to_api_dir(self) -> None:
        self._clean_directory(self.api_dir)
        for item in self.tmp_dir.iterdir():
            dest = self.api_dir / item.relative_to(self.tmp_dir)
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_func = shutil.copytree if item.is_dir() else shutil.copy2
            print(f"Copying {'directory' if item.is_dir() else 'file'} {item} to {dest}")
            copy_func(item, dest)

    def split_ref_file(self, encoding: str) -> None:
        try:
            self.tmp_dir.mkdir(exist_ok=True)
            for output_dir, symbols in self._process_files():
                if symbols:
                    for name, content in symbols.items():
                        (output_dir / f"{name}.md").write_text(content, encoding)

            self._move_generated_files_to_api_dir()
        finally:
            shutil.rmtree(self.tmp_dir)


def main() -> None:
    with windows_encoding() as encoding:
        website_dir = Path(__file__).parent.absolute()

        parser = argparse.ArgumentParser(description="Process API reference documentation")
        parser.add_argument(
            "--api-dir",
            type=Path,
            help="Directory containing API documentation to process",
            default=website_dir / "reference",
        )

        args = parser.parse_args()

        if args.api_dir.exists():
            # Force delete the directory and its contents
            shutil.rmtree(args.api_dir, ignore_errors=True)

        target_dir = args.api_dir.resolve().relative_to(website_dir)
        template_dir = website_dir / "mako_templates"

        # Generate API reference documentation
        print("Generating API reference documentation...")
        generate(target_dir, template_dir, encoding)

        # Split the API reference from submodules into separate files for each symbols
        symbol_files_generator = SplitReferenceFilesBySymbols(target_dir)
        symbol_files_generator.split_ref_file(encoding)

        # Convert MD to MDX
        print("Converting MD files to MDX...")
        convert_md_to_mdx(args.api_dir, encoding)

        # Create mint.json from the template file
        mint_json_template_path = website_dir / "mint-json-template.json.jinja"
        mint_json_path = website_dir / "mint.json"

        print("Generating mint.json from template...")
        generate_mint_json_from_template(mint_json_template_path, mint_json_path, encoding)

        # Update mint.json
        update_mint_json_with_api_nav(website_dir, args.api_dir, encoding)

        print("API reference processing complete!")


if __name__ == "__main__":
    main()
