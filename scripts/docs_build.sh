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
            python ./generate_api_references.py --force
        else
            python ./generate_api_references.py
        fi && \
        python ./process_notebooks.py render
}

docs_build() {
    local force=${1:-""}  # Default to empty string if no argument
    docs_generate "$force"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build "$1"
fi
