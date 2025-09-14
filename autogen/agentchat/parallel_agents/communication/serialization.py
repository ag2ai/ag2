# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pickle
from typing import Any

__all__ = ["MessageSerializer"]


class MessageSerializer:
    """Handles serialization/deserialization of messages for IPC"""

    def __init__(self):
        pass

    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes using pickle"""
        try:
            return pickle.dumps(data)
        except Exception as e:
            print(f"Serialization failed: {e}")
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes back to original data using pickle"""
        try:
            return pickle.loads(data)
        except Exception as e:
            print(f"Deserialization failed: {e}")
            raise
