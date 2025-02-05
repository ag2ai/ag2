#!/usr/bin/env python


# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import toml
from jinja2 import Template


def get_optional_dependencies(pyproject_path: str) -> dict:
    with open(pyproject_path) as f:
        pyproject_data = toml.load(f)

    optional_dependencies = pyproject_data.get("project", {}).get("optional-dependencies", {})
    return optional_dependencies


# Example usage
pyproject_path = Path(__file__).parent.joinpath("../pyproject.toml")
optional_dependencies = get_optional_dependencies(pyproject_path)
optional_groups = [group for group in optional_dependencies]

# for group, dependencies in optional_dependencies.items():
#     print(f"Group: {group}")
#     for dependency in dependencies:
#         print(f"  - {dependency}")

template_path = Path(__file__).parents[1].joinpath("setup.jinja")
assert template_path.exists()

with template_path.open("r") as f:
    template_str = f.read()

if len(template_str) < 100:
    raise ValueError("Template string is too short")

# Create a Jinja2 template object
template = Template(template_str)

for name in ["ag2", "autogen"]:
    file_name = f"setup_{name}.py"
    file_path = Path(__file__).parents[1].joinpath(file_name)
    # Render the template with the optional dependencies
    rendered_setup_py = template.render(optional_dependencies=optional_dependencies, name=name)

    # Write the rendered setup.py to a file
    with file_path.open("w") as setup_file:
        setup_file.write(rendered_setup_py)
