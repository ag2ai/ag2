#!/usr/bin/env bash

# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

coverage report -m --include="autogen/*" --omit="autogen/extensions/tmp_code_*.py"
