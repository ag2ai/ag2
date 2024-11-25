# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
import re
import sys
from pathlib import Path

REQUIRED_ELEMENTS = [r"Copyright.*Owners of https://github\.com/ag2ai", r"SPDX-License-Identifier: Apache-2\.0"]


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
    for py_file in Path(".").rglob("*.py"):
        if not should_check_file(py_file):
            continue

        missing_elements = check_file_header(py_file)
        if missing_elements:
            failed = True
            print(f"\nIncomplete or missing license header in: {py_file}")
            print(
                "\nSee https://ag2ai.github.io/ag2/docs/contributor-guide/contributing/#license-headers for guidance."
            )

            """
            # For more detailed output:
            print("Missing required elements:")
            for element in missing_elements:
                print(f"  - {element}")
            print("\nHeader should contain:")
            print("  1. Copyright notice with 'Owners of https://github.com/ag2ai'")
            print("  2. SPDX-License-Identifier: Apache-2.0")
            """

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
