#!/usr/bin/env bash
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

set -e
set -x

docs_generate() {
    local force=$1  # Get force flag as argument
    cd website && \
        # Only add --force if argument is exactly "--force"
        if [ "$force" = "--force" ]; then
            python3 ./generate_api_references.py --force
        else
            python3 ./generate_api_references.py
        fi
        # Note: process_notebooks.py render is not needed for the main docs build.
        # It's only required for mkdocs builds, which is handled separately in docs_build_mkdocs.sh
        # python3 ./process_notebooks.py render
}

docs_build() {
    local force=${1:-""}  # Default to empty string if no argument
    docs_generate "$force"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build "$1"
fi
