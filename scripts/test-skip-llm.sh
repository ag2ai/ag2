#!/usr/bin/env bash

# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

bash scripts/test.sh -m "not (openai or gemini or anthropic)" "$@"
