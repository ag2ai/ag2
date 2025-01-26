#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc, AG2AI OSS project maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

pytest --ff -vv --durations=10 --durations-min=1.0 "$@"
