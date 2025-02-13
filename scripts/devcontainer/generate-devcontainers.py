# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# A script to generate devcontainer files for different python versions

import shutil
from pathlib import Path

from jinja2 import Template

# List of python versions to generate devcontainer files for
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]
# Default python version for the main devcontainer
DEFAULT = "3.9"

DEVCONTAINER_JSON_TEMPLATE = Path("scripts/devcontainer/templates/devcontainer.json.jinja")


def generate_devcontainer_json_file(python_version: str) -> None:
    print(f"Generating devcontainer.json for python {python_version}")

    with open(DEVCONTAINER_JSON_TEMPLATE) as f:
        content = f.read()

    # Replace python_version with the current version using jinja template
    try:
        template = Template(content)
    except Exception as e:
        print(f"Error loading template: {e}")
        exit(1)

    data = {
        "python_version": python_version,
    }
    devcontainer_content = template.render(data)

    file_dir = (
        Path("./.devcontainer/").resolve()
        if python_version == DEFAULT
        else Path(f"./.devcontainer/python-{python_version}/").resolve()
    )
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_dir / "devcontainer.json", "w") as f:
        f.write(devcontainer_content + "\n")


def generate_devcontainer_files() -> None:
    for python_version in PYTHON_VERSIONS:
        # Delete existing devcontainer files
        files_to_delete = []
        if python_version == DEFAULT:
            files_to_delete = [Path("./.devcontainer/devcontainer.json")]

        files_to_delete += [
            Path(f"./.devcontainer/python-{python_version}/devcontainer.json"),
            Path(f"./.devcontainer/python-{python_version}/"),
        ]

        for file in files_to_delete:
            if file.exists():
                print(f"Deleting {file}")
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)

        generate_devcontainer_json_file(python_version)


if __name__ == "__main__":
    generate_devcontainer_files()
