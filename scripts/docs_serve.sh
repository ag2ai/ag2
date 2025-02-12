#!/usr/bin/env bash
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


set -e
set -x

# Store the force flag argument (default to empty string if not provided)
FORCE=${1:-""}

# Source the docs_build.sh script from the same directory
source "$(dirname "$0")/docs_build.sh"

# Run the docs_build function from docs_build.sh with the force flag
docs_build "$FORCE"

cd build
# Check if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Running npm install..."
    npm install
else
    echo "node_modules already exists, skipping npm install..."
fi

# Add the command to serve the documentation
echo "Serving documentation..."
npm run mintlify:dev
