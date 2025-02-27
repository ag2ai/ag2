#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

base_filter="not (openai or openai_realtime or gemini or gemini_realtime or anthropic or deepseek or ollama or bedrock or cerebras)"
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

base_filter="(not aux_neg_flag) or ($base_filter)"

echo $base_filter

bash scripts/test.sh -m "$base_filter" "${args[@]}"
