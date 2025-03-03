#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

llm_filter="not (openai or openai_realtime or gemini or gemini_realtime or anthropic or deepseek or ollama or bedrock or cerebras)"
optional_deps_filter="not (redis or docker or docs or rag or jupyter_executor or retrievechat or retrievechat_pgvector or retrievechat_mongodb or retrievechat_qdrant or graph_rag_falkor_db or neo4j or twilio or interop or browser_use or crawl4ai or websockets or commsagent_discord or commsagent_slack or commsagent-telegram or lmm)"
base_filter="$llm_filter and $optional_deps_filter"
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
