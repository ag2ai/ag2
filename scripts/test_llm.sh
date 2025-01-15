# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

# Default mark if none is provided
DEFAULT_MARK="openai or gemini"

# Initialize MARK as the default value
MARK="$DEFAULT_MARK"

# Parse arguments for the -m flag
while [[ $# -gt 0 ]]; do
  case $1 in
    -m)
      MARK="$2"  # Set MARK to the provided value
      shift 2     # Remove -m and its value from arguments
      ;;
    *)
      break  # If no more flags, stop processing options
      ;;
  esac
done

# Call the test script with the correct mark and any remaining arguments
bash scripts/test.sh "$@" -m "$MARK" --ignore=test/agentchat/contrib
