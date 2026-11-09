# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Run AG2 code and shell tools in caller-owned, persistent Sprites."""

from ag2.exceptions import missing_additional_dependency

try:
    from .environment import SpritesEnvironment
    from .sandbox import SpritesSandbox
except ImportError as e:
    SpritesEnvironment = missing_additional_dependency("SpritesEnvironment", "sprites-py>=0.7.2,<1", e)  # type: ignore[misc]
    SpritesSandbox = missing_additional_dependency("SpritesSandbox", "sprites-py>=0.7.2,<1", e)  # type: ignore[misc]

__all__ = ("SpritesEnvironment", "SpritesSandbox")
