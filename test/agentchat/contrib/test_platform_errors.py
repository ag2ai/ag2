import pytest
from autogen.agentchat.contrib.comms.platform_errors import (
    PlatformError,
    PlatformAuthenticationError,
    PlatformConnectionError,
    PlatformRateLimitError,
    PlatformTimeoutError,
)

def test_platform_error_base():
    error = PlatformError(message="Test error", platform_name="TestPlatform")
    assert str(error) == "[TestPlatform] Test error"
    assert error.platform_name == "TestPlatform"
    assert error.message == "Test error"

def test_platform_auth_error():
    error = PlatformAuthenticationError(
        message="Authentication failed",
        platform_name="TestPlatform"
    )
    assert isinstance(error, PlatformError)
    assert str(error) == "[TestPlatform] Authentication failed"
    assert error.platform_name == "TestPlatform"

def test_platform_connection_error():
    error = PlatformConnectionError(
        message="Connection failed",
        platform_name="TestPlatform"
    )
    assert isinstance(error, PlatformError)
    assert str(error) == "[TestPlatform] Connection failed"
    assert error.platform_name == "TestPlatform"

def test_platform_rate_limit_error():
    error = PlatformRateLimitError(
        message="Rate limit exceeded",
        platform_name="TestPlatform",
        retry_after=5
    )
    assert isinstance(error, PlatformError)
    assert str(error) == "[TestPlatform] Rate limit exceeded"
    assert error.platform_name == "TestPlatform"
    assert error.retry_after == 5

def test_platform_timeout_error():
    error = PlatformTimeoutError(
        message="Operation timed out",
        platform_name="TestPlatform"
    )
    assert isinstance(error, PlatformError)
    assert str(error) == "[TestPlatform] Operation timed out"
    assert error.platform_name == "TestPlatform"

def test_platform_error_inheritance():
    auth_error = PlatformAuthenticationError(message="Auth failed", platform_name="TestPlatform")
    conn_error = PlatformConnectionError(message="Connection failed", platform_name="TestPlatform")
    rate_error = PlatformRateLimitError(message="Rate limit", platform_name="TestPlatform", retry_after=5)
    timeout_error = PlatformTimeoutError(message="Timeout", platform_name="TestPlatform")

    assert isinstance(auth_error, PlatformError)
    assert isinstance(conn_error, PlatformError)
    assert isinstance(rate_error, PlatformError)
    assert isinstance(timeout_error, PlatformError)

def test_platform_error_custom_attributes():
    rate_error = PlatformRateLimitError(
        message="Rate limit",
        platform_name="TestPlatform",
        retry_after=10
    )
    assert hasattr(rate_error, "retry_after")
    assert rate_error.retry_after == 10

    with pytest.raises(AttributeError):
        base_error = PlatformError("Test", "TestPlatform")
        base_error.retry_after
