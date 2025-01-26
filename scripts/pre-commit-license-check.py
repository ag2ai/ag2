# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List

REQUIRED_ELEMENTS = [r"Copyright.*Owners of https://github\.com/ag2ai", r"SPDX-License-Identifier: Apache-2\.0"]


def get_github_pr_files() -> List[Path]:
    """Get list of Python files changed in a GitHub PR."""
    try:
        if os.getenv("GITHUB_EVENT_PATH"):
            with open(os.getenv("GITHUB_EVENT_PATH")) as f:
                event = json.load(f)

            # For pull requests, get changed files from the event payload
            if os.getenv("GITHUB_EVENT_NAME") == "pull_request":
                changed_files = []
                for file in event.get("pull_request", {}).get("changed_files", []):
                    filename = file.get("filename", "")
                    if filename.endswith(".py"):
                        changed_files.append(Path(filename))
                return changed_files

            # For push events, use git diff
            else:
                result = subprocess.run(
                    ["git", "diff", "--name-only", "HEAD^", "HEAD"], capture_output=True, text=True, check=True
                )
                return [Path(file) for file in result.stdout.splitlines() if file.endswith(".py")]
    except Exception as e:
        print(f"Error getting PR files: {e}")
    return []


def get_staged_files() -> List[Path]:
    """Get list of staged Python files using git command."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=AMR"], capture_output=True, text=True, check=True
        )
        files = result.stdout.splitlines()
        return [Path(file) for file in files if file.endswith(".py")]
    except subprocess.CalledProcessError as e:
        print(f"Error getting staged files: {e}")
        return []


def should_check_file(file_path: Path) -> bool:
    """Skip __init__.py files and check if file exists."""
    return file_path.name != "__init__.py" and file_path.exists()


def check_file_header(file_path: Path) -> List[str]:
    """Check if file has required license headers."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read(500)
            missing_elements = []
            for pattern in REQUIRED_ELEMENTS:
                if not re.search(pattern, content[:500], re.IGNORECASE):
                    missing_elements.append(pattern)
            return missing_elements
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []


def get_files_to_check() -> List[Path]:
    """Determine which files to check based on environment."""
    try:
        if "--all-files" in sys.argv:
            return list(Path().rglob("*.py"))

        if os.getenv("GITHUB_ACTIONS") == "true":
            return get_github_pr_files()

        return get_staged_files()
    except Exception as e:
        print(f"Error getting files to check: {e}")
        return []


def main() -> None:
    """Main function to check license headers."""
    try:
        failed = False
        files_to_check = get_files_to_check()

        if not files_to_check:
            print("No Python files to check")
            return

        for py_file in files_to_check:
            if not should_check_file(py_file):
                continue

            missing_elements = check_file_header(py_file)
            if missing_elements:
                failed = True
                print(f"\nIncomplete or missing license header in: {py_file}")
                print("\nSee https://docs.ag2.ai/docs/contributor-guide/contributing/#license-headers for guidance.")

        sys.exit(1 if failed else 0)
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
