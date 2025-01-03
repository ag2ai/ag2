# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash

pytest test --ignore=test/agentchat/contrib --skip-openai --durations=10 --durations-min=1.0
