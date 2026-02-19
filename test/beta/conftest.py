from unittest.mock import AsyncMock

import pytest


@pytest.fixture()
def async_mock() -> AsyncMock:
    return AsyncMock()
