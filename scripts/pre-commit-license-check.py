# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_ELEMENTS = [r"Copyright.*Owners of https://github\.com/ag2ai", r"SPDX-License-Identifier: Apache-2\.0"]


def get_changed_files():
    """Get list of Python files changed in this PR/push."""
    try:
        # If running in GitHub Actions PR
        if os.getenv("GITHUB_EVENT_PATH"):
            with open(os.getenv("GITHUB_EVENT_PATH")) as f:
                event = json.load(f)

            # For pull requests
            if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
                # Use the files listed in the PR event
                changed_files = []
                for file in event["pull_request"]["changed_files"]:
                    filename = file.get("filename", "")
                    if filename.endswith(".py"):
                        changed_files.append(Path(filename))
                return changed_files
            # For pushes
            else:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD^", "HEAD"], capture_output=True, text=True, check=True
                )
        # If running locally, check staged files
        else:
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=AMR"],
                capture_output=True,
                text=True,
                check=True,
            )

        # Filter for Python files and convert to Path objects
        return [Path(file) for file in result.stdout.splitlines() if file.endswith(".py")]
    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing files: {e}")
        sys.exit(1)


def should_check_file(file_path: Path) -> bool:
    # Skip __init__.py files
    return file_path.name != "__init__.py"


def check_file_header(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        # Read first few lines of the file
        content = f.read(500)

        # Check if all required elements are present near the start of the file
        missing_elements = []
        for pattern in REQUIRED_ELEMENTS:
            if not re.search(pattern, content[:500], re.IGNORECASE):
                missing_elements.append(pattern)

        return missing_elements


def main():
    failed = False
    changed_files = get_changed_files()

    if not changed_files:
        print("No Python files were changed.")
        return

    for py_file in changed_files:
        if not should_check_file(py_file):
            continue

        if not py_file.exists():
            print(f"Warning: File {py_file} no longer exists (may have been deleted)")
            continue

        missing_elements = check_file_header(py_file)
        if missing_elements:
            failed = True
            print(f"\nIncomplete or missing license header in: {py_file}")
            print(
                "\nSee https://ag2ai.github.io/ag2/docs/contributor-guide/contributing/#license-headers for guidance."
            )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
