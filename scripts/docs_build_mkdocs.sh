#!/usr/bin/env bash

set -e
set -x

if [ "$1" == "--force" ]; then
    cd website/mkdocs; python docs.py build --force
else
    cd website/mkdocs; python docs.py build
fi
