#!/usr/bin/env bash
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


set -e
set -x

docs_generate() {
    cd website && \
        python ./generate_api_references.py && \
        python ./process_notebooks.py render
}

docs_build() {
    docs_generate
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build
fi
