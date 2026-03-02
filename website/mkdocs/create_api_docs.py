# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import shutil
from importlib import import_module
from pathlib import Path

from autogen._website.generate_api_references import import_submodules
from autogen.doc_utils import get_target_module

API_META = "# 0.5 - API\n# 2 - Release\n# 3 - Contributing\n# 5 - Template Page\n# 10 - Default\nsearch:\n  boost: 0.5"

MD_API_META = "---\n" + API_META + "\n---\n\n"


def _is_private(name: str) -> bool:
    parts = name.split(".")
    return any(part.startswith("_") for part in parts)


def _merge_lists(members: list[str], submodules: list[str]) -> list[str]:
    members_copy = members[:]
    for sm in submodules:
        for i, el in enumerate(members_copy):
            if el.startswith(sm):
                members_copy.insert(i, sm)
                break
    return members_copy


def _add_all_submodules(members: list[str]) -> list[str]:
    def _f(x: str) -> list[str]:
        xs = x.split(".")
        return [".".join(xs[:i]) + "." for i in range(1, len(xs))]

    def _get_sorting_key(item):
        y = item.split(".")
        z = [f"~{a}" for a in y[:-1]] + [y[-1]]
        return ".".join(z)

    submodules = list(set(itertools.chain(*[_f(x) for x in members])))
    members = _merge_lists(members, submodules)
    members = list(dict.fromkeys(members))
    return sorted(members, key=_get_sorting_key)


def _resolve_case_collisions(members: list[str]) -> dict[str, str]:
    """Build a mapping from member name to file path, resolving case collisions.

    On case-insensitive filesystems (macOS), Tool.md and tool.md collide.
    When a collision is detected, the lowercase member gets a _func suffix.
    """
    path_map: dict[str, str] = {}  # member -> file path (without .md)
    seen_lower: dict[str, str] = {}  # lowercased path -> first member

    for x in members:
        if x.endswith("."):
            continue
        xs = x.split(".")
        file_path = "/".join(xs)
        lower_path = file_path.lower()

        if lower_path in seen_lower:
            # Case collision - disambiguate the lowercase one
            existing = seen_lower[lower_path]
            existing_last = existing.split(".")[-1]
            current_last = xs[-1]
            if current_last[0].islower():
                # Current is lowercase, add suffix
                file_path = "/".join(xs[:-1]) + f"/{xs[-1]}_func"
            elif existing_last[0].islower():
                # Existing was lowercase, update it
                path_map[existing] = "/".join(existing.split(".")[:-1]) + f"/{existing_last}_func"
        seen_lower[lower_path] = x
        path_map[x] = file_path

    return path_map


# Module-level cache populated during summary generation
_MEMBER_PATH_MAP: dict[str, str] = {}


def _get_api_summary_item(x: str) -> str:
    xs = x.split(".")
    if x.endswith("."):
        indent = " " * (4 * (len(xs) - 1))
        return f"{indent}- {xs[-2]}"
    else:
        indent = " " * (4 * (len(xs)))
        file_path = _MEMBER_PATH_MAP.get(x, "/".join(xs))
        return f"{indent}- [{xs[-1]}](docs/api-reference/{file_path}.md)"


def _get_api_summary(members: list[str]) -> str:
    global _MEMBER_PATH_MAP
    _MEMBER_PATH_MAP = _resolve_case_collisions(members)
    return "\n".join([_get_api_summary_item(x) for x in members])


def _generate_api_doc(name: str, docs_path: Path) -> Path:
    xs = name.split(".")
    module_name = ".".join(xs[:-1])
    member_name = xs[-1]
    file_path = _MEMBER_PATH_MAP.get(name, "/".join(xs))
    path = docs_path / f"{file_path}.md"
    content = f"::: {module_name}.{member_name}\n"

    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(MD_API_META + content)

    return path


def _generate_api_docs(members: list[str], docs_path: Path) -> list[Path]:
    return [_generate_api_doc(x, docs_path) for x in members if not x.endswith(".")]


def _filter_submodules_by_export_path(submodules: list[str], module_name: str) -> list[str]:
    ret_val = []
    for submodule in submodules:
        # Skip submodules not in the target package
        if not submodule.startswith(module_name):
            continue

        module = import_module(submodule)  # nosemgrep
        all = module.__all__ if hasattr(module, "__all__") else None
        for name, obj in module.__dict__.items():
            if not all:
                continue

            if all and name not in all:
                continue

            if not hasattr(obj, "__name__") or _is_private(name):
                continue

            target_module = get_target_module(obj)
            if target_module:
                if submodule == target_module:
                    ret_val.append(f"{submodule}.{obj.__name__}")
            else:
                fqn = f"{obj.__module__}.{obj.__name__}"
                # Only include if the module is part of our package
                if fqn.startswith(module_name):
                    ret_val.append(fqn)
    return ret_val


def _generate_api_docs_for_module(docs_path: Path, module_name: str) -> str:
    """Generate API documentation for a module.

    Args:
        docs_path: Path to mkdocs/docs directory.
        module_name: The name of the module.

    Returns:
        A string containing the API documentation for the module.

    """
    submodules = import_submodules(module_name)
    filtered_submodules = _filter_submodules_by_export_path(submodules, module_name)
    members_with_submodules = _add_all_submodules(filtered_submodules)

    api_summary = _get_api_summary(members_with_submodules)

    api_root = docs_path / "docs" / "api-reference"
    shutil.rmtree(api_root, ignore_errors=True)
    api_root.mkdir(parents=True, exist_ok=True)

    (api_root / ".meta.yml").write_text(API_META)

    _generate_api_docs(members_with_submodules, api_root)

    return api_summary


def create_api_docs(
    root_path: Path,
    module: str,
) -> None:
    """Generate API documentation for a module.

    Args:
        root_path: The root path of the project.
        module: The name of the module.

    """
    docs_dir = root_path / "docs"

    api = _generate_api_docs_for_module(docs_dir, module)

    # read summary template from file
    navigation_template = (docs_dir / "navigation_template.txt").read_text()

    summary = navigation_template.format(api=api)

    summary = "\n".join(filter(bool, (x.rstrip() for x in summary.split("\n"))))

    (docs_dir / "SUMMARY.md").write_text(summary)


def on_page_markdown(markdown, *, page, config, files):
    """Mkdocs hook to update the edit URL for the public API pages."""
    if page.edit_url and "public_api" in page.edit_url:
        page.edit_url = page.edit_url.replace("public_api", "api-reference")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    create_api_docs(root, "autogen")
