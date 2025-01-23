#!/usr/bin/env bash

set -e
set -x

docs_generate() {
    cd website && \
        python ./add_api_reference_to_docs.py && \
        python ./process_notebooks.py render
}

# install_packages() {
#     pip install -e ".[docs]"
# }

docs_build() {
    # install_packages && \
    docs_generate
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    docs_build
fi
