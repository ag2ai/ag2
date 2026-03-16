# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os

# Opt-in flag for recording prompt/tool content in OTel spans.
# Default OFF to prevent accidental secret/PII exfiltration to trace backends.
RECORD_CONTENT = os.environ.get("AG2_OTEL_RECORD_CONTENT", "").lower() in ("1", "true", "yes")
