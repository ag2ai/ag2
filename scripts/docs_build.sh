#!/usr/bin/env bash

set -e
set -x

# Function to build documentation
docs_build() {
    cd website &&
        python ./process_api_reference.py &&
        python ./process_notebooks.py render
}

# Execute the function only if the script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build
fi
