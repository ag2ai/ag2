# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import logging
from functools import wraps
from typing import Callable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def deprecated_by(
    new_class: type[BaseModel],
    param_mapping: dict[str, str] = None,
    default_params: dict[str, any] = None,
) -> Callable[[type[BaseModel]], Callable[..., BaseModel]]:
    param_mapping = param_mapping or {}
    default_params = default_params or {}

    def decorator(
        old_class: type[BaseModel],
        param_mapping: dict[str, str] = param_mapping,
        default_params: dict[str, any] = default_params,
    ) -> Callable[..., BaseModel]:
        @wraps(old_class)
        def wrapper(*args, **kwargs) -> BaseModel:
            logger.warning(
                f"{old_class.__name__} is deprecated by {new_class.__name__}. Please import it from {new_class.__module__} and use it instead."
            )
            # Translate old parameters to new parameters
            new_kwargs = {param_mapping.get(k, k): v for k, v in kwargs.items()}

            # Add default parameters if not already present
            for key, value in default_params.items():
                if key not in new_kwargs:
                    new_kwargs[key] = value

            # Pass the translated parameters to the new class
            return new_class(*args, **new_kwargs)

        return wrapper

    return decorator
