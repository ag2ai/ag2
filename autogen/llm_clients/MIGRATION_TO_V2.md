# Migration Guide: ModelClient V1 to ModelClientV2

This guide provides a comprehensive plan for migrating from the legacy ModelClient interface to the new ModelClientV2 interface with rich UnifiedResponse support.

## Table of Contents
- [Overview](#overview)
- [Why Migrate?](#why-migrate)
- [Architecture Comparison](#architecture-comparison)
- [Migration Strategy](#migration-strategy)
- [Step-by-Step Migration](#step-by-step-migration)
- [Backward Compatibility](#backward-compatibility)
- [Provider-Specific Considerations](#provider-specific-considerations)
- [Testing Strategy](#testing-strategy)
- [FAQ](#faq)

## Overview

ModelClientV2 introduces a new protocol for LLM clients that returns rich, provider-agnostic responses (UnifiedResponse) while maintaining backward compatibility with the existing ChatCompletion-based interface.

### Key Changes
- **Rich Response Format**: Returns `UnifiedResponse` with typed content blocks instead of flattened `ChatCompletion`
- **Direct Content Access**: Use `response.text`, `response.reasoning`, etc. instead of `message_retrieval()`
- **Forward Compatible**: Handles unknown content types via `GenericContent`
- **Dual Interface**: Supports both V2 (rich) and V1 (legacy) responses

## Why Migrate?

### Benefits of ModelClientV2

1. **Rich Content Support**: Access reasoning blocks, citations, multimodality, and other provider-specific features
2. **Provider Agnostic**: Unified format across OpenAI, Anthropic, Gemini, and other providers
3. **Type Safety**: Typed content blocks with enum-based content types
4. **Forward Compatibility**: Handles new content types without code changes
5. **Better Developer Experience**: Direct property access instead of parsing nested structures

### Example: Before vs After

**Before (ModelClient V1):**
```python
# V1 - Flattened ChatCompletion format
response = client.create(params)
messages = client.message_retrieval(response)
content = messages[0] if messages else ""

# Reasoning/thinking tokens lost or require provider-specific parsing
if hasattr(response, 'choices') and hasattr(response.choices[0], 'message'):
    if hasattr(response.choices[0].message, 'reasoning'):
        reasoning = response.choices[0].message.reasoning  # Provider-specific
```

**After (ModelClientV2):**
```python
# V2 - Rich UnifiedResponse format
response = client.create(params)
content = response.text                    # Direct access
reasoning = response.reasoning              # Rich content preserved
citations = response.get_content_by_type("citation")

# Access individual messages with typed content blocks
for message in response.messages:
    for content_block in message.content:
        if isinstance(content_block, ReasoningContent):
            print(f"Reasoning: {content_block.reasoning}")
        elif isinstance(content_block, TextContent):
            print(f"Text: {content_block.text}")
```

## Architecture Comparison

### ModelClient V1 (Legacy)

```python
class ModelClient(Protocol):
    def create(self, params: dict[str, Any]) -> ModelClientResponseProtocol:
        """Returns ChatCompletion-like response"""
        ...

    def message_retrieval(self, response) -> list[str]:
        """Extracts text content from response"""
        ...

    def cost(self, response) -> float: ...
    def get_usage(self, response) -> dict[str, Any]: ...
```

**Response Format:**
```python
ChatCompletion(
    id="...",
    model="...",
    choices=[
        Choice(
            message=Message(
                role="assistant",
                content="Plain text only"  # Rich content flattened
            )
        )
    ]
)
```

### ModelClientV2 (New)

```python
class ModelClientV2(Protocol):
    def create(self, params: dict[str, Any]) -> UnifiedResponse:
        """Returns rich UnifiedResponse"""
        ...

    def create_v1_compatible(self, params: dict[str, Any]) -> Any:
        """Backward compatibility method"""
        ...

    def cost(self, response: UnifiedResponse) -> float: ...
    def get_usage(self, response: UnifiedResponse) -> dict[str, Any]: ...
    # No message_retrieval - use response.text or response.messages directly
```

**Response Format:**
```python
UnifiedResponse(
    id="...",
    model="...",
    provider="openai",
    messages=[
        UnifiedMessage(
            role="assistant",
            content=[
                TextContent(type="text", text="Main response"),
                ReasoningContent(type="reasoning", reasoning="Let me think..."),
                CitationContent(type="citation", url="...", title="...", snippet="...")
            ]
        )
    ],
    usage={"prompt_tokens": 10, "completion_tokens": 20},
    cost=0.001
)
```

## Migration Strategy

### Phase 1: Implement Dual Interface (Current)
**Status**: ✅ Completed for OpenAICompletionsClient

**Goal**: Add V2 interface while maintaining V1 compatibility

```python
class OpenAICompletionsClient(ModelClient):  # Inherits V1 protocol
    """Implements V2 interface via duck typing"""

    def create(self, params: dict[str, Any]) -> UnifiedResponse:  # V2 method
        """Returns rich UnifiedResponse"""
        ...

    def message_retrieval(self, response: UnifiedResponse) -> list[str]:  # V1 compat
        """Flattens UnifiedResponse to text for legacy code"""
        return [msg.get_text() for msg in response.messages]

    def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:  # V2 compat
        """Converts UnifiedResponse to ChatCompletion format"""
        response = self.create(params)
        return self._to_chat_completion(response)
```

### Phase 2: Update OpenAIWrapper (Next)
**Status**: 🔄 In Progress

**Goal**: Support both V1 and V2 clients in routing layer

```python
class OpenAIWrapper:
    def create(self, params: dict[str, Any]) -> ModelClientResponseProtocol | UnifiedResponse:
        """Returns appropriate response type based on client"""
        client = self._clients[self._config_list_index]

        # Detect V2 clients by checking return type
        response = client.create(params)

        if isinstance(response, UnifiedResponse):
            # V2 client - rich response
            response._client = client  # Store client reference
            return response
        else:
            # V1 client - legacy response
            return response

    def extract_text_or_completion_object(self, response):
        """Handle both V1 and V2 responses"""
        if isinstance(response, UnifiedResponse):
            # V2 - use direct access
            return response.text
        else:
            # V1 - use message_retrieval
            client = self._response_metadata[response.id]["client"]
            return client.message_retrieval(response)
```

### Phase 3: Migrate Other Providers (Planned)
**Status**: 📋 Planned

**Priority Order:**
1. ✅ OpenAI (Completed - OpenAICompletionsClient)
2. 🔄 Gemini (High Priority - complex multimodal support)
3. 📋 Anthropic (High Priority - thinking tokens, citations)
4. 📋 Bedrock (Medium Priority - supports multiple models)
5. 📋 Together.AI, Groq, Mistral (Lower Priority - simpler APIs)

### Phase 4: Update Agent Layer (Future)
**Status**: 📋 Planned

**Goal**: Enable agents to consume rich content directly

```python
class ConversableAgent:
    def _generate_oai_reply_from_client(self, llm_client, messages, cache, agent):
        response = llm_client.create(params)

        if isinstance(response, UnifiedResponse):
            # V2 - process rich content
            self._process_reasoning(response.reasoning)
            self._process_citations(response.get_content_by_type("citation"))
            return response.text
        else:
            # V1 - legacy processing
            extracted = self.client.message_retrieval(response)
            return extracted
```

### Phase 5: Deprecation (Long-term)
**Status**: 📋 Planned

## Step-by-Step Migration

### For Client Implementers

#### Step 1: Inherit ModelClient (Maintain Compatibility)
```python
from autogen.llm_config.client import ModelClient

class MyProviderClient(ModelClient):
    """Inherit V1 protocol for OpenAIWrapper compatibility"""
    pass
```

#### Step 2: Implement V2 create() Method
```python
from autogen.llm_clients.models import (
    UnifiedResponse, UnifiedMessage, TextContent, ReasoningContent
)

def create(self, params: dict[str, Any]) -> UnifiedResponse:  # type: ignore[override]
    """Override with rich return type"""

    # 1. Call provider API
    raw_response = self._call_provider_api(params)

    # 2. Transform to UnifiedResponse with rich content blocks
    messages = []
    for choice in raw_response.choices:
        content_blocks = []

        # Extract text content
        if choice.message.content:
            content_blocks.append(TextContent(
                type="text",
                text=choice.message.content
            ))

        # Extract reasoning (provider-specific)
        if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
            content_blocks.append(ReasoningContent(
                type="reasoning",
                reasoning=choice.message.reasoning
            ))

        # Add more content types as needed...

        messages.append(UnifiedMessage(
            role=choice.message.role,
            content=content_blocks
        ))

    # 3. Create UnifiedResponse
    return UnifiedResponse(
        id=raw_response.id,
        model=raw_response.model,
        provider="my_provider",
        messages=messages,
        usage={
            "prompt_tokens": raw_response.usage.prompt_tokens,
            "completion_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        },
        cost=self._calculate_cost(raw_response)
    )
```

#### Step 3: Maintain V1 message_retrieval()
```python
def message_retrieval(self, response: UnifiedResponse) -> list[str]:  # type: ignore[override]
    """Flatten to text for V1 compatibility"""
    return [msg.get_text() for msg in response.messages]
```

#### Step 4: Implement create_v1_compatible()
```python
def create_v1_compatible(self, params: dict[str, Any]) -> dict[str, Any]:
    """Convert rich response to legacy format"""
    response = self.create(params)

    # Convert UnifiedResponse to ChatCompletion-like dict
    return {
        "id": response.id,
        "model": response.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": msg.role,
                    "content": msg.get_text()
                },
                "finish_reason": "stop"
            }
            for msg in response.messages
        ],
        "usage": response.usage
    }
```

#### Step 5: Update cost() and get_usage()
```python
def cost(self, response: UnifiedResponse) -> float:
    """Extract cost from UnifiedResponse"""
    return response.cost or 0.0

@staticmethod
def get_usage(response: UnifiedResponse) -> dict[str, Any]:
    """Extract usage from UnifiedResponse"""
    return {
        "prompt_tokens": response.usage.get("prompt_tokens", 0),
        "completion_tokens": response.usage.get("completion_tokens", 0),
        "total_tokens": response.usage.get("total_tokens", 0),
        "cost": response.cost or 0.0,
        "model": response.model
    }
```

### For Client Users

#### Step 1: Check Response Type
```python
response = llm_client.create(params)

if isinstance(response, UnifiedResponse):
    # V2 client - use rich interface
    text = response.text
    reasoning = response.reasoning
else:
    # V1 client - use legacy interface
    text = llm_client.message_retrieval(response)
```

#### Step 2: Use Direct Property Access
```python
# Instead of:
messages = client.message_retrieval(response)
content = messages[0] if messages else ""

# Use:
content = response.text  # Direct access
```

#### Step 3: Access Rich Content
```python
# Reasoning blocks (OpenAI o1/o3, Anthropic thinking)
if response.reasoning:
    print(f"Chain of thought: {response.reasoning}")

# Citations (web search, RAG)
citations = response.get_content_by_type("citation")
for citation in citations:
    print(f"Source: {citation.url} - {citation.title}")

# Images (Gemini, DALL-E)
images = response.get_content_by_type("image")
for image in images:
    print(f"Generated image: {image.image_url or image.data_uri}")
```

#### Step 4: Handle Unknown Content Types
```python
# Forward compatibility with GenericContent
for message in response.messages:
    for content_block in message.content:
        if isinstance(content_block, GenericContent):
            print(f"Unknown type: {content_block.type}")
            # Access fields dynamically
            all_fields = content_block.get_all_fields()
            print(f"Fields: {all_fields}")
```

## Backward Compatibility

### OpenAIWrapper Integration

**Current State**: OpenAIWrapper supports V2 clients through duck typing

```python
# OpenAIWrapper calls client.create() - works with both V1 and V2
response = client.create(params)

# For text extraction, OpenAIWrapper detects response type
if hasattr(response, 'text'):  # UnifiedResponse
    text = response.text
elif hasattr(response, 'message_retrieval_function'):  # V1 with stored metadata
    text = response.message_retrieval_function(response)
```

### Agent Compatibility

**Current State**: Agents work with both V1 and V2 clients

```python
# ConversableAgent._generate_oai_reply_from_client()
response = llm_client.create(params)

# Extract text - works with both formats
extracted_response = self.client.extract_text_or_completion_object(response)
```

**No Breaking Changes**: Existing agents continue to work without modifications

## Provider-Specific Considerations

### OpenAI (Completed ✅)
**Implementation**: `OpenAICompletionsClient`

**Supported Content Types:**
- `TextContent` - Standard text responses
- `ReasoningContent` - O1/O3 reasoning tokens
- `ToolCallContent` - Function/tool calls
- `ImageContent` - DALL-E generated images (future)

**Special Handling:**
- O1 models: Reasoning tokens extracted to ReasoningContent
- Streaming: Not yet implemented for V2 (uses V1 compatibility)
- Azure: Works through same client with different base_url

### Gemini (High Priority 🔄)
**Complexity**: High - extensive multimodal support

**Supported Content Types:**
- `TextContent` - Text responses
- `ImageContent` - Generated images (Imagen integration)
- `AudioContent` - Generated audio (future)
- `VideoContent` - Video understanding inputs
- `ToolCallContent` - Function calling

**Migration Challenges:**
- Complex content part structure (text, inline_data, file_data)
- Multiple generation modes (generateContent, generateImages)
- Safety ratings and finish reasons
- Grounding metadata and citations

**Recommended Approach:**
```python
class GeminiStatelessClient(ModelClient):
    def create(self, params: dict[str, Any]) -> UnifiedResponse:
        # Detect generation type from params
        if "generation_type" in params and params["generation_type"] == "image":
            return self._create_image_generation(params)
        else:
            return self._create_text_generation(params)

    def _create_text_generation(self, params) -> UnifiedResponse:
        # Convert OAI messages to Gemini format
        contents = oai_messages_to_gemini_messages(params["messages"])

        # Call Gemini API
        response = self.client.models.generate_content(
            model=params["model"],
            contents=contents,
            config=self._build_generation_config(params)
        )

        # Transform to UnifiedResponse
        return self._to_unified_response(response)

    def _to_unified_response(self, gemini_response) -> UnifiedResponse:
        messages = []
        for candidate in gemini_response.candidates:
            content_blocks = []

            # Extract text parts
            for part in candidate.content.parts:
                if part.text:
                    content_blocks.append(TextContent(type="text", text=part.text))
                elif part.inline_data:
                    # Handle inline images/audio
                    mime_type = part.inline_data.mime_type
                    if mime_type.startswith("image/"):
                        content_blocks.append(ImageContent(
                            type="image",
                            data_uri=f"data:{mime_type};base64,{part.inline_data.data}"
                        ))

            messages.append(UnifiedMessage(
                role=self._normalize_role(candidate.content.role),
                content=content_blocks
            ))

        return UnifiedResponse(
            id=f"gemini-{uuid.uuid4()}",
            model=gemini_response.model_name,
            provider="gemini",
            messages=messages,
            usage={
                "prompt_tokens": gemini_response.usage_metadata.prompt_token_count,
                "completion_tokens": gemini_response.usage_metadata.candidates_token_count,
                "total_tokens": gemini_response.usage_metadata.total_token_count
            },
            cost=self._calculate_cost(gemini_response)
        )
```

### Anthropic (High Priority 📋)
**Complexity**: Medium - thinking tokens and citations

**Supported Content Types:**
- `TextContent` - Standard responses
- `ReasoningContent` - Extended thinking mode
- `CitationContent` - Grounded responses with sources
- `ToolCallContent` - Tool use

**Migration Approach:**
```python
class AnthropicClient(ModelClient):
    def _to_unified_response(self, anthropic_response) -> UnifiedResponse:
        content_blocks = []

        # Extract text content
        for block in anthropic_response.content:
            if block.type == "text":
                content_blocks.append(TextContent(type="text", text=block.text))
            elif block.type == "thinking":
                content_blocks.append(ReasoningContent(
                    type="reasoning",
                    reasoning=block.thinking
                ))
            elif block.type == "tool_use":
                content_blocks.append(ToolCallContent(
                    type="tool_call",
                    id=block.id,
                    name=block.name,
                    arguments=json.dumps(block.input)
                ))

        messages = [UnifiedMessage(role="assistant", content=content_blocks)]

        return UnifiedResponse(
            id=anthropic_response.id,
            model=anthropic_response.model,
            provider="anthropic",
            messages=messages,
            usage={
                "prompt_tokens": anthropic_response.usage.input_tokens,
                "completion_tokens": anthropic_response.usage.output_tokens,
                "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens
            },
            cost=self._calculate_cost(anthropic_response)
        )
```

### Bedrock (Medium Priority 📋)
**Complexity**: Medium - wraps multiple providers

**Challenge**: Different underlying models (Claude, Llama, etc.) with different response formats

**Approach**: Detect model family and delegate to appropriate transformer

## Testing Strategy

### Unit Tests

**Test V2 Protocol Compliance:**
```python
def test_v2_protocol_compliance():
    """Verify client implements ModelClientV2 interface"""
    client = MyProviderClient()

    # Check required methods exist
    assert hasattr(client, "create")
    assert hasattr(client, "create_v1_compatible")
    assert hasattr(client, "cost")
    assert hasattr(client, "get_usage")

    # Check return types
    response = client.create({"model": "test", "messages": [...]})
    assert isinstance(response, UnifiedResponse)

    v1_response = client.create_v1_compatible({"model": "test", "messages": [...]})
    assert isinstance(v1_response, dict)
    assert "choices" in v1_response
```

**Test Rich Content Extraction:**
```python
def test_rich_content_types():
    """Verify all content types are properly extracted"""
    client = MyProviderClient()
    response = client.create(params)

    # Test direct access
    assert isinstance(response.text, str)
    assert response.text != ""

    # Test content type filtering
    reasoning_blocks = response.get_content_by_type("reasoning")
    assert all(isinstance(b, ReasoningContent) for b in reasoning_blocks)

    citations = response.get_content_by_type("citation")
    assert all(isinstance(c, CitationContent) for c in citations)
```

**Test V1 Compatibility:**
```python
def test_v1_backward_compatibility():
    """Verify V1 interface still works"""
    client = MyProviderClient()

    # V1 method should work
    response = client.create(params)
    messages = client.message_retrieval(response)
    assert isinstance(messages, list)
    assert all(isinstance(m, str) for m in messages)

    # V1 compatible response
    v1_response = client.create_v1_compatible(params)
    assert "choices" in v1_response
    assert "message" in v1_response["choices"][0]
```

### Integration Tests

**Test with OpenAIWrapper:**
```python
def test_openai_wrapper_integration():
    """Verify V2 client works with OpenAIWrapper"""
    config_list = [{
        "model": "gpt-4o",
        "api_key": "test",
        "api_type": "openai"
    }]

    wrapper = OpenAIWrapper(config_list=config_list)
    response = wrapper.create({"messages": [{"role": "user", "content": "Hello"}]})

    # Should return UnifiedResponse
    assert isinstance(response, UnifiedResponse)
    assert response.text
```

**Test with ConversableAgent:**
```python
def test_agent_integration():
    """Verify agents work with V2 clients"""
    agent = ConversableAgent(
        name="assistant",
        llm_config={"model": "gpt-4o", "api_key": "test"}
    )

    # Agent should handle V2 responses transparently
    reply = agent.generate_reply(
        messages=[{"role": "user", "content": "Hello"}]
    )

    assert isinstance(reply, str)
    assert reply != ""
```

### Test Fixtures

**Create reusable fixtures for testing:**
```python
@pytest.fixture
def mock_v2_response():
    """Mock UnifiedResponse for testing"""
    return UnifiedResponse(
        id="test-123",
        model="test-model",
        provider="test",
        messages=[
            UnifiedMessage(
                role="assistant",
                content=[
                    TextContent(type="text", text="Test response"),
                    ReasoningContent(type="reasoning", reasoning="Test reasoning")
                ]
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        cost=0.001
    )

@pytest.fixture
def mock_v2_client(mock_v2_response):
    """Mock V2 client for testing"""
    class MockV2Client:
        def create(self, params):
            return mock_v2_response

        def create_v1_compatible(self, params):
            return {
                "id": mock_v2_response.id,
                "model": mock_v2_response.model,
                "choices": [{"message": {"content": mock_v2_response.text}}]
            }

        def cost(self, response):
            return response.cost

        @staticmethod
        def get_usage(response):
            return {
                "prompt_tokens": response.usage["prompt_tokens"],
                "completion_tokens": response.usage["completion_tokens"],
                "total_tokens": response.usage["total_tokens"],
                "cost": response.cost,
                "model": response.model
            }

    return MockV2Client()
```

## FAQ

### Q: Do I need to migrate my existing client immediately?
**A:** No. V1 clients will continue to work indefinitely. Migration is only needed if you want to support rich content features.

### Q: Can I use V1 and V2 clients together?
**A:** Yes. OpenAIWrapper supports both simultaneously and handles routing automatically.

### Q: What if my provider doesn't support rich content?
**A:** You can still migrate - just return `UnifiedResponse` with only `TextContent` blocks. This provides API consistency even without rich features.

### Q: How do I handle streaming with V2?
**A:** Streaming support for V2 is planned. For now, use `create_v1_compatible()` for streaming use cases.

### Q: Will this break my existing agents?
**A:** No. All changes are backward compatible. Agents will automatically work with both V1 and V2 clients.

### Q: How do I test V2 clients without API keys?
**A:** Use the provided test fixtures (`mock_v2_client`, `mock_v2_response`) for unit tests. Integration tests can use `credentials_responses_*` fixtures that work without actual API calls.

### Q: What's the performance impact?
**A:** Minimal. UnifiedResponse is lightweight and most overhead is in API calls themselves. The rich content structure is lazy-evaluated where possible.

### Q: Can I add custom content types?
**A:** Yes! Use `ContentParser.register()` to add custom content types, or use `GenericContent` for one-off cases.

### Q: How do I migrate provider-specific features?
**A:** Use the `extra` field in content blocks or add fields to `GenericContent`:
```python
GenericContent(
    type="custom_provider_feature",
    custom_field="value",
    another_field=123
)
```

## Support and Feedback

- **Documentation**: See [ModelClientV2 protocol](/autogen/llm_clients/client_v2.py)
- **Examples**: Check [test_client_v2.py](/test/llm_clients/test_client_v2.py)
- **Issues**: Report migration issues on GitHub with `[v2-migration]` tag
- **Community**: Discuss migration strategies in AG2 Discord #client-development channel

---

**Last Updated**: 2025-11-13
**Version**: 1.0
**Status**: Phase 1 Complete (OpenAI), Phase 2 In Progress
