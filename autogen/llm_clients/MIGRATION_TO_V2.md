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
2. **Provider Agnostic**: Unified format across OpenAI, Anthropic (V2), Gemini, and other providers
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
**Status**: ✅ Completed for OpenAICompletionsClient and AnthropicV2Client

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

class AnthropicV2Client(ModelClient):  # V2 protocol, exposed as api_type: "anthropic_v2"
    """Supports rich UnifiedResponse replies (see Provider-Specific Considerations below)"""
    ...
```

### Phase 2: Update OpenAIWrapper (Now Complete)
**Status**: ✅

Both OpenAICompletionsClient and AnthropicV2Client are fully supported.

### Phase 3: Migrate Other Providers (Planned)
**Status**: 📋 Planned

**Priority Order:**
1. ✅ OpenAI (Completed - OpenAICompletionsClient)
2. 🔄 Gemini (High Priority - complex multimodal support)
3. ✅ Anthropic V2 (Completed - see below)
4. 📋 Bedrock (Medium Priority - supports multiple models)
5. 📋 Together.AI, Groq, Mistral (Lower Priority - simpler APIs)

### Phase 4: Update Agent Layer (Future)
**Status**: 📋 Planned

### Phase 5: Deprecation (Long-term)
**Status**: 📋 Planned

## Step-by-Step Migration

### For Anthropic V2 Users (NEW)
#### Step 1: Update config to use "api_type": "anthropic_v2"
```python
from autogen import LLMConfig, AssistantAgent
import os

llm_config = LLMConfig(
    config_list={
        "model": "claude-sonnet-4-5",
        "api_type": "anthropic_v2",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
)
agent = AssistantAgent("assistant", llm_config=llm_config)
```
#### Step 2: Use response_format for structured outputs (optional)
```python
from pydantic import BaseModel

class MySchema(BaseModel):
    ...

llm_config = LLMConfig(..., response_format=MySchema)
```
#### Step 3: Use UnifiedResponse interface as elsewhere (see OpenAI example). Access `.text`, `.reasoning`, or content blocks. For tool/type-safe usage, Anthropic V2 also supports "strict" tools and vision content.

#### Step 4: For backward compatibility with V1, use `create_v1_compatible()` or access `message_retrieval()` for legacy code.

### For Client Implementers
[...existing step-by-step instructions unchanged...]

## Backward Compatibility
[...unchanged...]

## Provider-Specific Considerations

### OpenAI (Completed ✅)
[...unchanged...]

### Gemini (High Priority 🔄)
[...unchanged...]

### Anthropic V2 (Completed ✅)
**Implementation**: `AnthropicV2Client`, `api_type: "anthropic_v2"`

**Supported Content Types:**
- `TextContent` - Standard responses
- `ReasoningContent` - Extended "thinking" mode with reasoning blocks
- `CitationContent` - (future/experimental for citations)
- `ToolCallContent` - Tool use and function calls (with strict mode)
- `GenericContent` - Plug-in for future unknown types
- `ImageContent` - Vision/image input and output (vision models)

**Special Handling:**
- V2 interface returns UnifiedResponse with all features mapped
- Native support for Pydantic models or JSON schema with response_format parameter (guaranteed schema compliance)
- Strict tool APIs ("strict": True in tool definitions) for strongly typed function/tool calling
- Full vision/multimodal support for Claude 3.5+ vision models
- Chain-of-thought/"thinking" blocks are mapped to ReasoningContent
- For fallback or legacy support (V1), use `create_v1_compatible()` for ChatCompletion-like output

**Migration Approach:**
```python
from autogen import AssistantAgent, LLMConfig
import os
from pydantic import BaseModel

class MyOutputModel(BaseModel):
    field: str

llm_config = LLMConfig(
    config_list={
        "model": "claude-sonnet-4-5",
        "api_type": "anthropic_v2",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
    },
    response_format=MyOutputModel,
)
agent = AssistantAgent("assistant", llm_config=llm_config)
# agent.run(...) returns UnifiedResponse with .text, .reasoning, etc.
```

**V1 to V2 Changes:**
- Update `api_type` in config from `anthropic` to `anthropic_v2`
- Use `response_format` with a Pydantic model or JSON schema for structured outputs
- Access `.text`, `.reasoning`, and content types on UnifiedResponse
- Tool definitions use `strict: True` for strong typing (see docs/demo notebook for example)
- For vision, ensure you use an image-capable Claude model (see Claude model docs)

**See:**
- [Anthropic V2 Client User Guide](/docs/user-guide/models/anthropic#anthropic-v2-client)
- [AgentChat Anthropic V2 Example Notebook](notebook/agentchat_anthropic_v2_client_example.ipynb)

### Bedrock (Medium Priority 📋)
[...unchanged...]

## Testing Strategy
[...unchanged...]

## FAQ

**Q: Can I migrate my Anthropic (V1) code to V2?**
A: Yes! Anthropic V2 is now a recommended upgrade for most use cases: just set `api_type: "anthropic_v2"` in your config and (optionally) use `response_format` for structured outputs or strict tool calls. Vision support and richer content is automatic for compatible models.

**Q: Is OpenAIWrapper/agent tooling compatible with Anthropic V2?**
A: Yes, all layering is now V2 compatible. Responses will be UnifiedResponse; use `.text`, `.reasoning`, etc. as with OpenAI V2.

**Q: What if my code expects V1's ChatCompletion?**
A: Use `create_v1_compatible()` or the legacy message_retrieval to maintain compatibility, or update your agent logic to use UnifiedResponse.

**Q: How do I get structured output schemas with Anthropic V2?**
A: Supply a `response_format` (Pydantic model or dict schema). Valid JSON responses matching the schema will be returned automatically, even with tool calls or image input.

**Q: What about vision/image input?**
A: Use a vision-capable Claude model (e.g. "claude-3-5-haiku-20241022") and format user messages with the image input specification as shown in the [notebook examples](notebook/agentchat_anthropic_v2_client_example.ipynb).

**Q: Can I still use Anthropic (V1)?**
A: Yes, but Anthropic V2 is recommended for new code or to benefit from strict outputs, improved tool/multimodal support, and richer content variety.

---

**Last Updated**: 2025-11-13
**Version**: 1.1
**Status**: Phase 1 Complete (OpenAI/Anthropic V2), Phase 2 In Progress
