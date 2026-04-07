#!/usr/bin/env bash

# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
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

echo "Running beta tests with mark: $MARK"

# Call the test script targeting only test/beta
bash scripts/test.sh test/beta "$@" -m "$MARK"
