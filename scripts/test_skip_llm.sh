# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

bash scripts/test.sh -m "not llm" --ignore=test/agentchat/contrib "$@"
