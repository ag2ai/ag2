# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""bhived shared-lesson extension for AG2.

Maintainer: yossufyahya2000
"""

from ag2.exceptions import missing_additional_dependency

try:
    from .toolkit import (
        BhivedFailedApproach,
        BhivedLesson,
        BhivedQueryResponse,
        BhivedToolkit,
        BhivedWarning,
        BhivedWriteResult,
    )
except ImportError as e:
    BhivedToolkit = missing_additional_dependency("BhivedToolkit", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]
    BhivedLesson = missing_additional_dependency("BhivedLesson", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]
    BhivedWarning = missing_additional_dependency("BhivedWarning", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]
    BhivedFailedApproach = missing_additional_dependency("BhivedFailedApproach", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]
    BhivedQueryResponse = missing_additional_dependency("BhivedQueryResponse", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]
    BhivedWriteResult = missing_additional_dependency("BhivedWriteResult", "bhived>=0.1.0,<0.2", e)  # type: ignore[misc]

__all__ = (
    "BhivedFailedApproach",
    "BhivedLesson",
    "BhivedQueryResponse",
    "BhivedToolkit",
    "BhivedWarning",
    "BhivedWriteResult",
)
