# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "Field",
]


class Field:
    """Represents a description field for use in type annotations.

    This class is used to store a description for an annotated field, often used for
    documenting or validating fields in a context or data model.
    """

    def __init__(self, description: str) -> None:
        """Initializes the Field with a description.

        Args:
            description: The description text for the field.
        """
        self._description = description

    @property
    def description(self) -> str:
        return self._description
