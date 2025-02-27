#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Default mark if none is provided
DEFAULT_MARK="openai or openai_realtime or gemini or gemini_realtime or anthropic or deepseek"

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

# MARK="(not aux_neg_flag) or ($MARK)"

echo "Running tests with mark: $MARK"

# Call the test script with the correct mark and any remaining arguments
bash scripts/test.sh "$@" -m "$MARK" --ignore=test/agentchat/contrib
