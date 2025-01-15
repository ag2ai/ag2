# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

# Default mark if none is provided
DEFAULT_MARK="openai or gemini"

# Use the provided mark or fallback to the default
MARK=${1:-"$DEFAULT_MARK"}

# Shift positional parameters so additional arguments can be passed
shift

# Call the test script with the mark
bash scripts/test.sh -m "$MARK" --ignore=test/agentchat/contrib "$@"
