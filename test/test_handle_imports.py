# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from autogen.handle_imports import check_for_missing_imports


def test_check_for_missing_imports():
    with check_for_missing_imports():
        pass  # Safe to attempt, even if it fails
    assert True
