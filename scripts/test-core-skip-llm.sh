#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Initialize a variable for yepcode-specific ignore flags
YEPCODE_IGNORE_FLAGS=""

# Get Python major and minor versions
PYTHON_MAJOR_VERSION=$(python -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR_VERSION=$(python -c 'import sys; print(sys.version_info.minor)')

# If the Python version is 3.10, set the flags to ignore the yepcode test files
if [ "$PYTHON_MAJOR_VERSION" -eq 3 ] && [ "$PYTHON_MINOR_VERSION" -eq 10 ]; then
    echo "Python 3.10 detected. Excluding yepcode tests to prevent import errors."
    YEPCODE_IGNORE_FLAGS="--ignore=test/coding/test_yepcode_executor.py --ignore=test/coding/test_yepcode_executor_integration.py"
fi

# Call the main test script, passing along the original ignore flag,
# the new conditional flags (which will be an empty string for other Python versions),
# and any other arguments passed to this script.
bash scripts/test-skip-llm.sh --ignore=test/agentchat/contrib $YEPCODE_IGNORE_FLAGS "$@"
