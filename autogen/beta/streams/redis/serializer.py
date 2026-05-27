# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import pickle
from enum import Enum
from typing import Any

from autogen.beta.events._serialization import deserialize_value, serialize_value


class Serializer(Enum):
    """Serialization format for Redis storage and pub/sub transport."""

    JSON = "json"  # default
    PICKLE = "pickle"


def serialize(obj: Any, fmt: Serializer) -> bytes:
    """Serialize an event to bytes using the specified format."""
    if fmt is Serializer.PICKLE:
        return pickle.dumps(obj)
    return json.dumps(serialize_value(obj)).encode()


def deserialize(data: bytes, fmt: Serializer) -> Any:
    """Deserialize bytes back to an event using the specified format."""
    if fmt is Serializer.PICKLE:
        return pickle.loads(data)  # noqa: S301
    return deserialize_value(json.loads(data))
