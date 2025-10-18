#!/usr/bin/env bash

set -e
set -x

cd website/mkdocs; python3 docs.py live "$@"
