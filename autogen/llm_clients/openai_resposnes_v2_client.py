# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


from typing import Any
from autogen.llm_clients.client_v2 import ModelClientV2
from autogen.llm_clients.models.unified_response import UnifiedResponse


class OpenAIResponsesV2Client(ModelClientV2):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        response_format: Any = None,
        workspace_dir: str | None = None,
        allowed_paths: list[str] | None = None,
        **kwargs: Any,
    ):
        """Initialize with state management and tool support."""



    def create(self, params: dict[str, Any]) -> UnifiedResponse:
          """Create completion with stateful conversation support."""
        # 1. Handle state (previous_response_id)
        # 2. Convert messages to input format
        # 3. Handle built-in tools
        # 4. Call API (create or parse)
        # 5. Transform to UnifiedResponse
        # 6. Update state
    
    def _transform_response(
        self, 
        response: Any, 
        model: str
    ) -> UnifiedResponse:
        """Transform Responses API Response to UnifiedResponse."""
        # Process output items:
        # - message → TextContent
        # - reasoning → ReasoningContent  
        # - function_call → ToolCallContent
        # - image_generation_call → ImageContent
        # - web_search_call → CitationContent
        # - apply_patch_call → GenericContent
    

    def _get_previous_response_id(self) -> str | None:
        """Get current conversation state."""
    
    def _set_previous_response_id(self, response_id: str | None) -> None:
        """Update conversation state."""
        
    def reset_conversation(self) -> None:
        """Reset conversation state (start new conversation)."""