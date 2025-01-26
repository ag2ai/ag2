#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai OSS project maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

base_filter="not (openai or gemini or anthropic or deepseek)"
args=()
while [[ $# -gt 0 ]]; do
	if [[ "$1" == "-m" ]]; then
		shift
		base_filter="$base_filter and ($1)"
	else
		args+=("$1")
	fi
	shift
done

bash scripts/test.sh -m "$base_filter" "${args[@]}"
