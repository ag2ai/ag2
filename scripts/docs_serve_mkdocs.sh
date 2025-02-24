#!/usr/bin/env bash

set -e
set -x

cd website/mkdocs; python docs.py live "$@"
