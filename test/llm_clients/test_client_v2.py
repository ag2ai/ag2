# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ModelClientV2 protocol."""

from typing import Any

from autogen.llm_clients import TextContent, UnifiedMessage, UnifiedResponse


class MockClientV2:
    """Mock implementation of ModelClientV2 for testing."""

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    def create(self, params: dict[str, Any]) -> UnifiedResponse:
        """Create a mock unified response."""
        message = UnifiedMessage(
            role="assistant",
            content=[TextContent(type="text", text="Mock response")],
        )
        return UnifiedResponse(
            id="mock-123",
            model=params.get("model", "mock-model"),
            provider="mock",
            messages=[message],
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            cost=0.001,
        )

    def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:
        """Create a mock v1 compatible response."""
        # Simplified mock of ChatCompletionExtended
        return {
            "id": "mock-123",
            "model": params.get("model", "mock-model"),
            "choices": [{"message": {"content": "Mock response"}}],
        }

    def cost(self, response: UnifiedResponse) -> float:
        """Return cost from response."""
        return response.cost or 0.0

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:
        """Extract usage from response."""
        return {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "cost": response.cost or 0.0,
            "model": response.model,
        }


class TestModelClientV2Protocol:
    """Test ModelClientV2 protocol compliance."""

    def test_protocol_compliance(self):
        """Test that MockClientV2 implements ModelClientV2 protocol."""
        client = MockClientV2()

        # Check required attributes
        assert hasattr(client, "RESPONSE_USAGE_KEYS")
        assert hasattr(client, "create")
        assert hasattr(client, "create_v1_compatible")
        assert hasattr(client, "cost")
        assert hasattr(client, "get_usage")

        # Check RESPONSE_USAGE_KEYS
        assert isinstance(client.RESPONSE_USAGE_KEYS, list)
        assert "prompt_tokens" in client.RESPONSE_USAGE_KEYS
        assert "completion_tokens" in client.RESPONSE_USAGE_KEYS
        assert "total_tokens" in client.RESPONSE_USAGE_KEYS
        assert "cost" in client.RESPONSE_USAGE_KEYS
        assert "model" in client.RESPONSE_USAGE_KEYS

    def test_create_method(self):
        """Test create() method returns UnifiedResponse."""
        client = MockClientV2()
        params = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

        response = client.create(params)

        assert isinstance(response, UnifiedResponse)
        assert response.id == "mock-123"
        assert response.model == "test-model"
        assert response.provider == "mock"
        assert len(response.messages) == 1

    def test_create_v1_compatible_method(self):
        """Test create_v1_compatible() returns legacy format."""
        client = MockClientV2()
        params = {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}

        response = client.create_v1_compatible(params)

        assert isinstance(response, dict)
        assert "id" in response
        assert "model" in response
        assert "choices" in response

    def test_cost_method(self):
        """Test cost() method."""
        client = MockClientV2()
        params = {"model": "test-model"}

        response = client.create(params)
        cost = client.cost(response)

        assert isinstance(cost, float)
        assert cost == 0.001

    def test_get_usage_method(self):
        """Test get_usage() static method."""
        client = MockClientV2()
        params = {"model": "test-model"}

        response = client.create(params)
        usage = client.get_usage(response)

        assert isinstance(usage, dict)
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert "cost" in usage
        assert "model" in usage
        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30
        assert usage["model"] == "test-model"

    def test_direct_content_access(self):
        """Test direct access to rich content via UnifiedResponse properties."""
        client = MockClientV2()
        params = {"model": "test-model"}

        response = client.create(params)

        # Direct access to text content via response property
        assert response.text == "Mock response"
        assert isinstance(response.messages, list)
        assert len(response.messages) == 1

        # Access message content directly
        message = response.messages[0]
        assert message.get_text() == "Mock response"


class TestModelClientV2DualInterface:
    """Test dual interface pattern (v2 and v1 compatibility)."""

    def test_create_returns_rich_response(self):
        """Test that create() returns rich UnifiedResponse."""
        client = MockClientV2()
        params = {"model": "test-model"}

        response = client.create(params)

        # Should be UnifiedResponse with all features
        assert isinstance(response, UnifiedResponse)
        assert hasattr(response, "messages")
        assert hasattr(response, "usage")
        assert hasattr(response, "cost")
        assert hasattr(response, "provider")

    def test_create_v1_compatible_flattens_response(self):
        """Test that create_v1_compatible() returns flattened response."""
        client = MockClientV2()
        params = {"model": "test-model"}

        response = client.create_v1_compatible(params)

        # Should be dict-like legacy format
        assert isinstance(response, dict)
        assert "choices" in response
        # Note: Real implementation would properly flatten UnifiedResponse

    def test_both_methods_use_same_params(self):
        """Test that both methods accept same parameters."""
        client = MockClientV2()
        params = {"model": "test-model", "temperature": 0.7, "max_tokens": 100}

        # Both methods should accept same params
        response_v2 = client.create(params)
        response_v1 = client.create_v1_compatible(params)

        assert response_v2.model == "test-model"
        assert response_v1["model"] == "test-model"


class TestModelClientV2UsageTracking:
    """Test usage tracking functionality."""

    def test_usage_includes_all_required_keys(self):
        """Test that usage dict includes all required keys."""
        client = MockClientV2()
        response = client.create({"model": "test"})
        usage = client.get_usage(response)

        for key in client.RESPONSE_USAGE_KEYS:
            assert key in usage

    def test_cost_calculation(self):
        """Test cost calculation from response."""
        client = MockClientV2()
        response = client.create({"model": "test"})

        cost = client.cost(response)
        usage_cost = client.get_usage(response)["cost"]

        assert cost == usage_cost
        assert cost == 0.001

    def test_token_counting(self):
        """Test token counting in usage."""
        client = MockClientV2()
        response = client.create({"model": "test"})
        usage = client.get_usage(response)

        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 20
        assert usage["total_tokens"] == 30


class TestModelClientV2ErrorHandling:
    """Test error handling in ModelClientV2 implementations."""

    def test_missing_cost_in_response(self):
        """Test handling when response has no cost."""

        class ClientNoCost:
            RESPONSE_USAGE_KEYS = MockClientV2.RESPONSE_USAGE_KEYS

            def create(self, params: dict[str, Any]) -> UnifiedResponse:
                message = UnifiedMessage(
                    role="assistant",
                    content=[TextContent(type="text", text="Response")],
                )
                return UnifiedResponse(
                    id="test",
                    model="test",
                    provider="test",
                    messages=[message],
                    cost=None,  # No cost
                )

            def cost(self, response: UnifiedResponse) -> float:
                return response.cost or 0.0

            @staticmethod
            def get_usage(response: UnifiedResponse) -> dict[str, Any]:
                return {"cost": response.cost or 0.0}

        client = ClientNoCost()
        response = client.create({})
        cost = client.cost(response)

        assert cost == 0.0

    def test_empty_messages(self):
        """Test handling responses with no messages."""

        class ClientEmptyMessages:
            def create(self, params: dict[str, Any]) -> UnifiedResponse:
                return UnifiedResponse(
                    id="test",
                    model="test",
                    provider="test",
                    messages=[],  # Empty messages
                )

        client = ClientEmptyMessages()
        response = client.create({})

        # Access content directly via response property
        assert response.text == ""
        assert len(response.messages) == 0


class TestModelClientV2Integration:
    """Integration tests for ModelClientV2."""

    def test_full_workflow(self):
        """Test complete workflow from create to usage extraction."""
        client = MockClientV2()

        # 1. Create request
        params = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
        }

        # 2. Get response
        response = client.create(params)
        assert isinstance(response, UnifiedResponse)

        # 3. Extract usage
        usage = client.get_usage(response)
        assert usage["model"] == "gpt-4"
        assert usage["total_tokens"] == 30

        # 4. Calculate cost
        cost = client.cost(response)
        assert cost == 0.001

        # 5. Access content directly via UnifiedResponse properties
        assert response.text == "Mock response"
        assert len(response.messages) == 1
        assert response.messages[0].get_text() == "Mock response"

    def test_backward_compatibility_workflow(self):
        """Test backward compatibility workflow."""
        client = MockClientV2()

        # Old code uses create_v1_compatible
        params = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        response = client.create_v1_compatible(params)

        # Should work with legacy code expecting dict format
        assert isinstance(response, dict)
        assert response["model"] == "gpt-4"
        assert "choices" in response

    def test_migration_path(self):
        """Test gradual migration from v1 to v2."""
        client = MockClientV2()
        params = {"model": "test"}

        # Old code
        old_response = client.create_v1_compatible(params)
        assert isinstance(old_response, dict)

        # New code
        new_response = client.create(params)
        assert isinstance(new_response, UnifiedResponse)

        # Both should return same basic information
        assert old_response["model"] == new_response.model
