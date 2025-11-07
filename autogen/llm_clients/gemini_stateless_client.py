# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""
Google Gemini Stateless API Client implementing ModelClient protocol.

This client uses the modern client.models.generate_content() API (NOT the legacy chat API)
and returns rich UnifiedResponse objects with:
- Thinking content (Gemini 2.5+ extended thinking via part.thought flag)
- Tool calls and function execution
- Code execution (Gemini-specific feature)
- Multimodal content (text, images, audio, video)
- Structured outputs

The client preserves all Gemini-specific features in UnifiedResponse format
and is compatible with AG2's agent system through ModelClient protocol.

Example:
    ```python
    from autogen.llm_clients import GeminiStatelessClient

    # API Key mode (Gemini Developer API)
    client = GeminiStatelessClient(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

    # Vertex AI mode (Google Cloud)
    client = GeminiStatelessClient(vertexai=True, project="my-gcp-project", location="us-central1")

    # Basic text generation
    response = client.create({
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Explain quantum computing"}],
    })
    print(response.text)

    # Thinking mode (Gemini 2.5+)
    response = client.create({
        "model": "gemini-2.5-pro",
        "messages": [{"role": "user", "content": "What is 15 factorial?"}],
        "thinking_config": {"include_thoughts": True},
    })
    for thinking in response.thinking:
        print(f"Thinking: {thinking.thinking}")
    ```
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import uuid
from typing import Any

from ..import_utils import optional_import_block
from ..llm_config.client import ModelClient
from .models import (
    AudioContent,
    GenericContent,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    ToolResultContent,
    UnifiedMessage,
    UnifiedResponse,
    VideoContent,
)

with optional_import_block():
    import vertexai
    from google import genai
    from google.auth.credentials import Credentials
    from google.genai.types import (
        Content,
        FunctionCall,
        FunctionDeclaration,
        FunctionResponse,
        GenerateContentConfig,
        GoogleSearch,
        Part,
        Schema,
        Tool,
        ToolCodeExecution,
        Type,
    )
    from vertexai.generative_models import Content as VertexAIContent
    from vertexai.generative_models import FunctionDeclaration as vaiFunctionDeclaration
    from vertexai.generative_models import GenerationConfig, GenerativeModel
    from vertexai.generative_models import Part as VertexAIPart
    from vertexai.generative_models import Tool as vaiTool

logger = logging.getLogger(__name__)


class GeminiStatelessClient(ModelClient):
    """
    Google Gemini Stateless API client implementing ModelClient protocol.

    Uses client.models.generate_content() API which supports:
    - Text generation
    - Multimodal content (images, audio, video)
    - Tool/function calling (automatic function calling)
    - Thinking/reasoning (Gemini 2.5+ models)
    - Code execution
    - Structured outputs
    - Safety settings

    Key Differences from Legacy Gemini Client:
    - Uses generate_content API instead of chat API
    - Returns UnifiedResponse instead of ChatCompletion
    - Preserves thinking content via ThinkingContent
    - Preserves code execution via GenericContent
    - Full multimodal output support

    Authentication:
    1. API Key mode (Gemini Developer API):
       - Simple, no GCP required
       - Set api_key parameter or GOOGLE_GEMINI_API_KEY env var

    2. Vertex AI mode (Google Cloud):
       - For enterprise GCP integration
       - Requires project, location, and credentials
       - Set vertexai=True
    """

    RESPONSE_USAGE_KEYS: list[str] = ["prompt_tokens", "completion_tokens", "total_tokens", "cost", "model"]

    # Parameter mapping from AG2 to Gemini
    PARAMS_MAPPING = {
        "max_tokens": "max_output_tokens",
        "seed": "seed",
        "stop_sequences": "stop_sequences",
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "max_output_tokens": "max_output_tokens",
    }

    def __init__(
        self,
        api_key: str | None = None,
        vertexai: bool = False,
        project: str | None = None,
        location: str | None = None,
        credentials: Credentials | None = None,
        timeout: float = 60.0,
        **kwargs: Any,
    ):
        """
        Initialize Gemini Stateless API client.

        Two authentication modes:

        1. API Key mode (Gemini Developer API):
           ```python
           client = GeminiStatelessClient(api_key="...")
           ```

        2. Vertex AI mode (Google Cloud):
           ```python
           client = GeminiStatelessClient(vertexai=True, project="my-gcp-project", location="us-central1")
           ```

        Args:
            api_key: Google Gemini API key (or set GOOGLE_GEMINI_API_KEY env var)
            vertexai: Use Vertex AI instead of Gemini Developer API
            project: GCP project ID (Vertex AI only)
            location: GCP compute location (Vertex AI only)
            credentials: google.auth.credentials.Credentials object (Vertex AI only)
            timeout: Request timeout in seconds
            **kwargs: Additional arguments

        Raises:
            ImportError: If google-genai package not installed
            AssertionError: If invalid parameter combination
        """
        # Try to get API key from parameter or environment
        if not api_key and not vertexai:
            api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

        # Determine authentication mode
        if api_key:
            # Gemini Developer API mode
            self.use_vertexai = False
            self.client = genai.Client(api_key=api_key)

            # Can't specify GCP parameters with API key
            if project or location:
                raise AssertionError("Google Cloud project and compute location cannot be set when using an API Key!")
        else:
            # Vertex AI mode
            self.use_vertexai = True
            self._initialize_vertexai(project, location, credentials)
            self.client = None  # Vertex AI uses GenerativeModel directly

        self.timeout = timeout
        self._response_format = None  # For structured outputs

    def _initialize_vertexai(
        self, project: str | None = None, location: str | None = None, credentials: Credentials | None = None
    ):
        """Initialize Vertex AI with project, location, and credentials."""
        vertexai_init_args = {}

        if project:
            vertexai_init_args["project"] = project
        if location:
            vertexai_init_args["location"] = location
        if credentials:
            if not isinstance(credentials, Credentials):
                raise AssertionError("Object type google.auth.credentials.Credentials is expected!")
            vertexai_init_args["credentials"] = credentials

        if vertexai_init_args:
            vertexai.init(**vertexai_init_args)

    def create(self, params: dict[str, Any]) -> UnifiedResponse:
        """
        Create completion using generate_content API.

        Args:
            params: Request parameters including:
                - model: Model name (e.g., "gemini-2.5-flash")
                - messages: List of message dicts
                - temperature: Sampling temperature
                - max_tokens: Maximum output tokens
                - tools: Tool/function definitions
                - thinking_config: Thinking mode configuration (Gemini 2.5+)
                - safety_settings: Safety settings
                - response_format: Structured output format

        Returns:
            UnifiedResponse with rich content blocks

        Example:
            ```python
            response = client.create({
                "model": "gemini-2.5-flash",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.7,
            })
            ```
        """
        model = params.get("model", "gemini-2.5-flash")
        messages = params.get("messages", [])

        # Extract system instruction (if first message is system)
        system_instruction = self._extract_system_instruction(messages)

        # Convert AG2 messages to Gemini Contents
        contents = self._ag2_messages_to_gemini_contents(messages)

        # Convert tools if provided (must be done before building config)
        tools = None
        if "tools" in params and params["tools"]:
            tools = self._convert_tools(params["tools"])

        # Build generation config (includes tools for Gemini Developer API)
        config = self._build_generation_config(
            params, system_instruction, tools=tools if not self.use_vertexai else None
        )

        # Call API
        try:
            if self.use_vertexai:
                # Vertex AI pattern
                generation_config_dict = {}
                if config.temperature is not None:
                    generation_config_dict["temperature"] = config.temperature
                if config.max_output_tokens is not None:
                    generation_config_dict["max_output_tokens"] = config.max_output_tokens
                if config.top_p is not None:
                    generation_config_dict["top_p"] = config.top_p
                if config.top_k is not None:
                    generation_config_dict["top_k"] = config.top_k

                model_obj = GenerativeModel(model, system_instruction=system_instruction)

                vertex_contents = []
                for content in contents:
                    vertex_parts = []
                    for part in content.parts:
                        if hasattr(part, "text") and part.text:
                            vertex_parts.append(VertexAIPart.from_text(part.text))
                        elif (
                            hasattr(part, "function_call")
                            and part.function_call
                            or hasattr(part, "function_response")
                            and part.function_response
                            or hasattr(part, "inline_data")
                            and part.inline_data
                        ):
                            vertex_parts.append(part)
                    vertex_contents.append(VertexAIContent(parts=vertex_parts, role=content.role))

                response = model_obj.generate_content(
                    vertex_contents,
                    generation_config=GenerationConfig(**generation_config_dict) if generation_config_dict else None,
                    tools=tools,
                )
            else:
                # Gemini Developer API pattern
                response = self.client.models.generate_content(model=model, contents=contents, config=config)

            # Transform to UnifiedResponse
            return self._transform_response(response, model)

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _extract_system_instruction(self, messages: list[dict[str, Any]]) -> str | None:
        """Extract system instruction from first message if present."""
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from multimodal content
                texts = [item["text"] for item in content if item.get("type") == "text"]
                return "\n".join(texts) if texts else None
        return None

    def _ag2_messages_to_gemini_contents(self, messages: list[dict[str, Any]]) -> list[Content]:
        """
        Convert AG2 message format to Gemini Content format.

        AG2 message: {"role": str, "content": str | list, "tool_calls": ..., ...}
        Gemini Content: Content(role=str, parts=[Part(...), ...])
        """
        contents = []

        # Skip system message (handled separately)
        start_idx = 1 if messages and messages[0].get("role") == "system" else 0

        for message in messages[start_idx:]:
            parts = []
            role = message.get("role", "user")

            # Handle text content
            content = message.get("content")
            if isinstance(content, str) and content:
                parts.append(Part(text=content))

            # Handle multimodal content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        parts.append(Part(text=item["text"]))
                    elif item.get("type") == "image_url":
                        image_part = self._convert_image_to_part(item)
                        if image_part:
                            parts.append(image_part)

            # Handle tool calls
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    func = tool_call.get("function", {})
                    try:
                        args = json.loads(func.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}

                    parts.append(Part(function_call=FunctionCall(name=func.get("name", ""), args=args)))

            # Handle tool responses
            if role == "tool":
                tool_name = message.get("name", "unknown")
                tool_content = message.get("content", "")
                parts.append(
                    Part(function_response=FunctionResponse(name=tool_name, response={"result": tool_content}))
                )

            # Map role to Gemini format
            gemini_role = "user" if role in ["user", "system", "tool"] else "model"

            if parts:
                contents.append(Content(role=gemini_role, parts=parts))

        # Ensure alternating user/model pattern (Gemini requirement)
        return self._ensure_alternating_messages(contents)

    def _ensure_alternating_messages(self, contents: list[Content]) -> list[Content]:
        """
        Ensure messages alternate between user and model roles.

        Gemini requires:
        1. First message must be from user
        2. Last message must be from user
        3. Messages must alternate between user and model
        """
        if not contents:
            return contents

        # Ensure first message is from user
        if contents[0].role != "user":
            contents.insert(0, Content(role="user", parts=[Part(text="start chat")]))

        # Ensure last message is from user
        if contents[-1].role != "user":
            contents.append(Content(role="user", parts=[Part(text="continue")]))

        # Ensure alternating pattern
        result = []
        prev_role = None
        for content in contents:
            if prev_role == content.role:
                # Insert dummy message to alternate
                dummy_role = "model" if content.role == "user" else "user"
                result.append(Content(role=dummy_role, parts=[Part(text="...")]))
            result.append(content)
            prev_role = content.role

        return result

    def _convert_image_to_part(self, image_item: dict) -> Part | None:
        """Convert AG2 image format to Gemini Part."""
        url = image_item.get("image_url", {}).get("url", "")

        if not url:
            return None

        try:
            if url.startswith("data:image"):
                # Data URI - extract base64
                match = re.match(r"data:image/(\\w+);base64,(.+)", url)
                if match:
                    mime_type, data = match.groups()
                    from google.genai.types import Blob

                    return Part(inline_data=Blob(mime_type=f"image/{mime_type}", data=base64.b64decode(data)))

            elif url.startswith("gs://"):
                # GCS URI (Vertex AI)
                from google.genai.types import FileData

                return Part(file_data=FileData(file_uri=url, mime_type="image/png"))

            elif url.startswith("http"):
                # Download and inline (not recommended for production)
                import requests

                response = requests.get(url, timeout=10)
                from google.genai.types import Blob

                return Part(inline_data=Blob(mime_type="image/png", data=response.content))
        except Exception as e:
            logger.warning(f"Failed to convert image: {e}")
            return None

        return None

    def _build_generation_config(
        self, params: dict[str, Any], system_instruction: str | None, tools: list[Tool] | None = None
    ) -> GenerateContentConfig:
        """Build Gemini GenerateContentConfig from AG2 parameters.

        Args:
            params: Request parameters
            system_instruction: System instruction string
            tools: Converted tool declarations (for Gemini Developer API)

        Returns:
            GenerateContentConfig with all parameters including tools
        """
        config_args = {}

        # Map standard parameters
        if "temperature" in params:
            config_args["temperature"] = params["temperature"]
        if "max_tokens" in params:
            config_args["max_output_tokens"] = params["max_tokens"]
        if "top_p" in params:
            config_args["top_p"] = params["top_p"]
        if "top_k" in params:
            config_args["top_k"] = params["top_k"]
        if "seed" in params:
            config_args["seed"] = params["seed"]
        if "stop_sequences" in params:
            config_args["stop_sequences"] = params["stop_sequences"]

        # System instruction
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        # Thinking config (Gemini 2.5+)
        if "thinking_config" in params:
            config_args["thinking_config"] = params["thinking_config"]

        # Safety settings
        if "safety_settings" in params:
            config_args["safety_settings"] = params["safety_settings"]

        # Structured outputs
        if "response_format" in params:
            response_format = params["response_format"]
            if isinstance(response_format, dict) and response_format.get("type") == "json_object":
                config_args["response_mime_type"] = "application/json"
                if "schema" in response_format:
                    config_args["response_json_schema"] = response_format["schema"]

        # Image generation config (gemini-2.5-flash-image)
        if "image_config" in params:
            config_args["image_config"] = params["image_config"]

        # Response modalities (for image-only or audio output)
        if "response_modalities" in params:
            config_args["response_modalities"] = params["response_modalities"]

        # Speech config (for TTS models)
        if "speech_config" in params:
            config_args["speech_config"] = params["speech_config"]

        # Tools (Gemini Developer API requires tools in config)
        if tools:
            config_args["tools"] = tools

        return GenerateContentConfig(**config_args)

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[Any]:
        """Convert AG2 tool format to Gemini Tool format.

        Supports:
        - Function tools: {"type": "function", "function": {...}}
        - Code execution: {"code_execution": {}}
        - Google Search: {"type": "google_search"}

        Returns:
            list[Tool] for Gemini Developer API or list[vaiTool] for Vertex AI
        """
        # Check for code execution tool
        for tool in tools:
            if "code_execution" in tool:
                # Both Gemini Developer API and Vertex AI use the same format with google.genai SDK
                return [Tool(code_execution=ToolCodeExecution())]

        # Check for built-in Google Search tool
        for tool in tools:
            if tool.get("type") == "google_search" and not self.use_vertexai:
                return [Tool(google_search=GoogleSearch())]

        # Convert function tools
        functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]

                if self.use_vertexai:
                    # Vertex AI format
                    function = vaiFunctionDeclaration(
                        name=func["name"],
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                else:
                    # Gemini Developer API format
                    function = self._create_gemini_function_declaration(func)

                functions.append(function)

        if self.use_vertexai:
            return [vaiTool(function_declarations=functions)]
        else:
            return [Tool(function_declarations=functions)]

    def _create_gemini_function_declaration(self, func: dict[str, Any]) -> FunctionDeclaration:
        """Create Gemini FunctionDeclaration from AG2 function format."""
        function_declaration = FunctionDeclaration()
        function_declaration.name = func["name"]
        function_declaration.description = func.get("description", "")

        if "parameters" in func and func["parameters"].get("properties"):
            function_declaration.parameters = self._create_gemini_schema(func["parameters"])

        return function_declaration

    def _create_gemini_schema(self, json_schema: dict[str, Any]) -> Schema:
        """Convert JSON Schema to Gemini Schema recursively."""
        schema = Schema()

        # Map type
        type_mapping = {
            "string": Type.STRING,
            "integer": Type.INTEGER,
            "number": Type.NUMBER,
            "object": Type.OBJECT,
            "array": Type.ARRAY,
            "boolean": Type.BOOLEAN,
        }

        schema_type = json_schema.get("type", "string")
        schema.type = type_mapping.get(schema_type, Type.STRING)

        # Handle object properties
        if schema_type == "object" and "properties" in json_schema:
            schema.properties = {}
            for prop_name, prop_schema in json_schema["properties"].items():
                schema.properties[prop_name] = self._create_gemini_schema(prop_schema)

            # Required fields
            if "required" in json_schema:
                schema.required = json_schema["required"]

        # Handle arrays
        if schema_type == "array" and "items" in json_schema:
            schema.items = self._create_gemini_schema(json_schema["items"])

        # Description
        if "description" in json_schema:
            schema.description = json_schema["description"]

        return schema

    def _transform_response(self, gemini_response: Any, model: str) -> UnifiedResponse:
        """
        Transform Gemini GenerateContentResponse to UnifiedResponse.

        Maps all Gemini content types to UnifiedResponse content blocks:
        - part.text → TextContent
        - part.thought=True → ThinkingContent
        - part.function_call → ToolCallContent
        - part.function_response → ToolResultContent
        - part.executable_code → GenericContent
        - part.code_execution_result → GenericContent
        - part.inline_data (images/audio/video) → ImageContent/AudioContent/VideoContent
        """
        messages = []

        # Extract candidates
        candidates = gemini_response.candidates if hasattr(gemini_response, "candidates") else []

        for candidate in candidates:
            content_blocks = []

            # Process all parts in the candidate's content
            content = candidate.content if hasattr(candidate, "content") else None
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    # Text content (not thinking)
                    if hasattr(part, "text") and part.text and not getattr(part, "thought", False):
                        content_blocks.append(TextContent(type="text", text=part.text))

                    # Thinking content (Gemini 2.5+)
                    elif hasattr(part, "thought") and part.thought and hasattr(part, "text") and part.text:
                        content_blocks.append(ThinkingContent(type="thinking", thinking=part.text))

                    # Tool calls
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        content_blocks.append(
                            ToolCallContent(
                                type="tool_call",
                                id=str(uuid.uuid4()),
                                name=fc.name,
                                arguments=json.dumps(dict(fc.args)) if hasattr(fc, "args") else "{}",
                            )
                        )

                    # Tool results
                    elif hasattr(part, "function_response") and part.function_response:
                        fr = part.function_response
                        content_blocks.append(
                            ToolResultContent(
                                type="tool_result",
                                tool_call_id="",  # Gemini doesn't provide tool_call_id
                                output=json.dumps(fr.response) if hasattr(fr, "response") else "",
                            )
                        )

                    # Code execution (Gemini-specific)
                    elif hasattr(part, "executable_code") and part.executable_code:
                        ec = part.executable_code
                        content_blocks.append(
                            GenericContent(
                                type="executable_code",
                                content={
                                    "language": getattr(ec, "language", "PYTHON"),
                                    "code": getattr(ec, "code", ""),
                                },
                            )
                        )

                    elif hasattr(part, "code_execution_result") and part.code_execution_result:
                        cer = part.code_execution_result
                        content_blocks.append(
                            GenericContent(
                                type="code_execution_result",
                                content={"outcome": getattr(cer, "outcome", ""), "output": getattr(cer, "output", "")},
                            )
                        )

                    # Multimodal content
                    elif hasattr(part, "inline_data") and part.inline_data:
                        inline_data = part.inline_data
                        mime_type = getattr(inline_data, "mime_type", "")

                        if mime_type.startswith("image/"):
                            # Convert to base64 data URI
                            data = getattr(inline_data, "data", b"")
                            if isinstance(data, bytes):
                                b64_data = base64.b64encode(data).decode("utf-8")
                                data_uri = f"data:{mime_type};base64,{b64_data}"
                                content_blocks.append(ImageContent(type="image", image_url=data_uri))

                        elif mime_type.startswith("audio/"):
                            data = getattr(inline_data, "data", b"")
                            if isinstance(data, bytes):
                                b64_data = base64.b64encode(data).decode("utf-8")
                                data_uri = f"data:{mime_type};base64,{b64_data}"
                                content_blocks.append(AudioContent(type="audio", audio_url=data_uri))

                        elif mime_type.startswith("video/"):
                            data = getattr(inline_data, "data", b"")
                            if isinstance(data, bytes):
                                b64_data = base64.b64encode(data).decode("utf-8")
                                data_uri = f"data:{mime_type};base64,{b64_data}"
                                content_blocks.append(VideoContent(type="video", video_url=data_uri))

            # Create unified message
            role = "assistant" if (content and getattr(content, "role", "model") == "model") else "user"
            messages.append(UnifiedMessage(role=role, content=content_blocks))

        # Extract usage (including thinking tokens for Gemini 2.5+)
        usage = {}
        thinking_tokens = 0
        if hasattr(gemini_response, "usage_metadata"):
            um = gemini_response.usage_metadata
            thinking_tokens = getattr(um, "thoughts_token_count", 0) or 0
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(um, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(um, "total_token_count", 0) or 0,
                "thinking_tokens": thinking_tokens,  # Gemini 2.5+ thinking mode
            }

        # Calculate cost (thinking tokens may be priced differently)
        # Note: For image/audio generation models, completion_tokens may be 0
        cost = self._calculate_cost(
            model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), thinking_tokens
        )

        # Extract finish reason
        finish_reason = None
        if candidates:
            finish_reason = self._map_finish_reason(getattr(candidates[0], "finish_reason", None))

        # Provider metadata
        provider_metadata = {}
        if hasattr(gemini_response, "model_version"):
            provider_metadata["model_version"] = gemini_response.model_version
        if hasattr(gemini_response, "prompt_feedback"):
            provider_metadata["prompt_feedback"] = str(gemini_response.prompt_feedback)
        if candidates and hasattr(candidates[0], "safety_ratings"):
            provider_metadata["safety_ratings"] = str(candidates[0].safety_ratings)

        return UnifiedResponse(
            id=getattr(gemini_response, "response_id", None) or str(uuid.uuid4()),
            model=model,
            provider="gemini",
            messages=messages,
            usage=usage,
            finish_reason=finish_reason,
            status="completed",
            provider_metadata=provider_metadata,
            cost=cost,
        )

    def _map_finish_reason(self, gemini_finish_reason: Any) -> str | None:
        """Map Gemini finish reason to standard format."""
        if not gemini_finish_reason:
            return None

        reason_str = str(gemini_finish_reason)

        # Map common reasons
        if "STOP" in reason_str:
            return "stop"
        elif "MAX_TOKENS" in reason_str or "LENGTH" in reason_str:
            return "length"
        elif "SAFETY" in reason_str or "RECITATION" in reason_str:
            return "content_filter"
        else:
            return "stop"

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int, thinking_tokens: int = 0) -> float:
        """
        Calculate cost for Gemini API call including thinking tokens.

        For Gemini 2.5+ models with thinking mode, thinking tokens may be priced
        differently ($3/million for some preview models).

        Args:
            model: Model name
            input_tokens: Prompt tokens
            output_tokens: Completion tokens (non-thinking)
            thinking_tokens: Thinking tokens (Gemini 2.5+ with thinking mode)
        """
        # Import the existing cost calculation function
        from ..oai.gemini import calculate_gemini_cost

        # Base cost for prompt + non-thinking completion
        base_cost = calculate_gemini_cost(self.use_vertexai, input_tokens, output_tokens, model)

        # Add thinking token cost (if applicable)
        # Note: According to gemini.py line 947, some Gemini 2.5 preview models
        # charge $3/million for thinking tokens
        thinking_cost = 0.0
        if thinking_tokens > 0:
            model_lower = model.lower()
            if "gemini-2.5-flash-preview-04-17" in model_lower or "gemini-2.5-flash-preview-05-20" in model_lower:
                # $3 per million thinking tokens
                thinking_cost = 3.0 * thinking_tokens / 1e6
            # Other Gemini 2.5+ models may have different thinking token pricing
            # Add more cases as pricing is announced

        return base_cost + thinking_cost

    def cost(self, response: UnifiedResponse) -> float:
        """Calculate cost from UnifiedResponse."""
        return response.cost if response.cost is not None else 0.0

    @staticmethod
    def get_usage(response: UnifiedResponse) -> dict[str, Any]:
        """Extract usage statistics from UnifiedResponse."""
        return {
            "prompt_tokens": response.usage.get("prompt_tokens", 0),
            "completion_tokens": response.usage.get("completion_tokens", 0),
            "total_tokens": response.usage.get("total_tokens", 0),
            "cost": response.cost,
            "model": response.model,
        }

    def message_retrieval(self, response: UnifiedResponse) -> list[str] | list[dict[str, Any]]:
        """
        Retrieve text content from response messages.

        For rich content (thinking, tool calls, images, audio, etc.), returns message dicts.
        For text-only responses, returns list of strings.
        """
        # Check if response contains only text
        has_non_text_content = False
        for message in response.messages:
            for block in message.content:
                if block.type not in ("text", "thinking"):
                    has_non_text_content = True
                    break
            if has_non_text_content:
                break

        # If response has images, audio, video, or tool calls, return full message dicts
        if has_non_text_content:
            messages = []
            for message in response.messages:
                # Convert UnifiedMessage to dict format for AG2 compatibility
                content_blocks = []
                for block in message.content:
                    if block.type == "text":
                        content_blocks.append({"type": "text", "text": block.text})
                    elif block.type == "image":
                        content_blocks.append({"type": "image_url", "image_url": {"url": block.image_url}})
                    elif block.type == "audio":
                        content_blocks.append({"type": "audio_url", "audio_url": {"url": block.audio_url}})
                    elif block.type == "video":
                        content_blocks.append({"type": "video_url", "video_url": {"url": block.video_url}})
                    elif block.type == "thinking":
                        content_blocks.append({"type": "text", "text": f"[Thinking: {block.thinking}]"})

                # For multimodal content, use list format; for empty, use empty string
                if content_blocks:
                    messages.append({"role": message.role, "content": content_blocks})
                else:
                    messages.append({"role": message.role, "content": ""})
            return messages

        # Text-only response: extract text strings
        texts = []
        for message in response.messages:
            for block in message.content:
                if block.type == "text":
                    texts.append(block.text)
                elif block.type == "thinking":
                    texts.append(f"[Thinking: {block.thinking}]")
        return texts


# LLMConfigEntry for Gemini Stateless Client
try:
    from typing import Literal

    from ..llm_config.entry import LLMConfigEntry

    class GeminiStatelessLLMConfigEntry(LLMConfigEntry):
        """Configuration entry for GeminiStatelessClient with ModelClientV2 architecture."""

        api_type: Literal["google_stateless"] = "google_stateless"

        # Gemini-specific optional parameters
        thinking_config: dict[str, Any] | None = None
        """Thinking mode configuration for Gemini 2.5+ models."""

        vertexai: bool | None = None
        """Whether to use Vertex AI mode instead of Gemini Developer API."""

        project: str | None = None
        """GCP project ID for Vertex AI mode."""

        location: str | None = None
        """GCP location for Vertex AI mode (e.g., 'us-central1')."""

        proxy: str | None = None
        """HTTP proxy for API requests."""

        # Media generation config (NEW)
        image_config: dict[str, Any] | None = None
        """Image generation configuration (aspect_ratio, etc.) for gemini-2.5-flash-image."""

        response_modalities: list[str] | None = None
        """Output modalities: ['Text', 'Image'] for image generation or ['AUDIO'] for TTS."""

        speech_config: dict[str, Any] | None = None
        """Text-to-speech configuration (voice_config, etc.) for TTS models."""

        def create_client(self) -> ModelClient:
            """Create GeminiStatelessClient from this configuration."""
            config_dict = self.model_dump(exclude={"api_type"}, exclude_none=True)
            return GeminiStatelessClient(**config_dict)

except ImportError:
    # If pydantic or llm_config not available, skip config entry class
    pass
