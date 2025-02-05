#!/usr/bin/env bash
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


set -e
set -x

# Source the docs_build.sh script from the same directory
source "$(dirname "$0")/docs_build.sh"

# Run the docs_build function from docs_build.sh
docs_build

# Install npm packages
echo "Running npm install..."
npm install

# Add the command to serve the documentation
echo "Serving documentation..."
npm run mintlify:dev
