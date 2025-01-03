#!/usr/bin/env bash

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
