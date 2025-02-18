#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

# Default values
OPTIONAL_DEPENDENCIES=""
LLM=""

# Parse the arguments using a while loop
while [[ $# -gt 0 ]]; do
  case $1 in
    -d|--optional-dependencies)   # handle both -d and --optional-dependencies
      OPTIONAL_DEPENDENCIES=$2
      shift 2
      ;;
    -l|--llm)                     # handle both -l and --llm
      LLM=$2
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Convert OPTIONAL_DEPENDENCIES to MARKER
MARKER="${OPTIONAL_DEPENDENCIES//-/_}"

# Prepare the command to be executed
COMMAND="scripts/test.sh -m \"($MARKER) and ($LLM)\""

# Echo the constructed command
echo "Executing command: bash -c $COMMAND"

# Run the constructed command directly
bash -c "$COMMAND"
