# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

# A script to generate devcontainer files for different python versions

import os
from pathlib import Path

from jinja2 import Template

# List of python versions to generate devcontainer files for
PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]
DEFAULT = "3.10"


def generate_docker_compose_file(python_version: str):
    print(f"Generating docker-compose.yml for python {python_version}")

    # Read file docker-compose.yml.jinja from ./devcontainer_templates/docker-compose.yml.jinja
    docker_compose_template = Path("scripts/devcontainer_templates/docker-compose.yml.jinja")
    with open(docker_compose_template, "r") as f:
        content = f.read()

    # Replace python_version with the current version using jinja template
    template = Template(content)
    data = {
        "python_version": python_version,
        "mount_volume": "../" if python_version == DEFAULT else "../../",
        "env_file": "./devcontainer.env" if python_version == DEFAULT else "../../devcontainer.env",
    }
    docker_compose_content = template.render(data)

    file_dir = (
        Path("./.devcontainer/") if python_version == DEFAULT else Path(f"./.devcontainer/python-{python_version}/")
    )
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_dir / "docker-compose.yml", "w") as f:
        f.write(docker_compose_content + "\n")


def generate_devcontainer_json_file(python_version: str):
    print(f"Generating devcontainer.json for python {python_version}")

    # Read file devcontainer.json.jinja from ./devcontainer_templates/devcontainer.json.jinja
    devcontainer_template = Path("scripts/devcontainer_templates/devcontainer.json.jinja")
    with open(devcontainer_template, "r") as f:
        content = f.read()

    # Replace python_version with the current version using jinja template
    template = Template(content)
    data = {
        "python_version": python_version,
    }
    devcontainer_content = template.render(data)

    file_dir = (
        Path("./.devcontainer/") if python_version == DEFAULT else Path(f"./.devcontainer/python-{python_version}/")
    )
    file_dir.mkdir(parents=True, exist_ok=True)

    with open(file_dir / "devcontainer.json", "w") as f:
        f.write(devcontainer_content + "\n")


def generate_devcontainer_files():
    for python_version in PYTHON_VERSIONS:
        generate_docker_compose_file(python_version)
        generate_devcontainer_json_file(python_version)


if __name__ == "__main__":
    generate_devcontainer_files()
