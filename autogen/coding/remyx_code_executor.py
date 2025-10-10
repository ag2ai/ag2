# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Remyx code executor implementation for research paper execution."""

import logging
import os
from collections.abc import Callable
from typing import ClassVar

from pydantic import Field

from ..doc_utils import export_module
from .base import CodeExecutor, CodeExtractor, CodeResult
from .docker_commandline_code_executor import DockerCommandLineCodeExecutor
from .markdown_code_extractor import MarkdownCodeExtractor

try:
    from dotenv import load_dotenv
    _load_dotenv: Callable[[], bool] | None = load_dotenv
except ImportError:
    _load_dotenv = None

# Import from remyxai package (installed as dependency)
try:
    from remyxai.api.search import Asset, get_asset as remyxai_get_asset
    from remyxai.client.search import SearchClient as RemyxSearchClient
except ImportError:
    Asset = None
    RemyxSearchClient = None
    remyxai_get_asset = None

logger = logging.getLogger(__name__)


@export_module("autogen.coding")
class RemyxCodeResult(CodeResult):
    """A code result class for Remyx executor."""

    arxiv_id: str | None = Field(default=None, description="The arXiv ID for this execution.")
    paper_title: str | None = Field(default=None, description="The paper title.")


@export_module("autogen.coding")
class RemyxCodeExecutor(DockerCommandLineCodeExecutor):
    """A code executor that runs research paper code in local Docker containers.

    This executor extends DockerCommandLineCodeExecutor to:
    1. Search and fetch paper metadata from Remyx API (via remyxai package)
    2. Pull paper-specific Docker images
    3. Execute code in pre-configured research environments

    All execution happens locally on the user's machine. The Remyx API (accessed via
    remyxai package) is only used for metadata discovery - no code is sent to remote servers.

    The executor supports research papers from the Remyx catalog that have
    Docker images with pre-installed dependencies and code.

    Args:
        arxiv_id (Optional[str]): arXiv ID to search and execute (e.g., "2010.11929v2").
            If provided, will fetch paper metadata and Docker image from Remyx API.
        image (Optional[str]): Docker image to use (overrides arxiv_id lookup).
        api_key (Optional[str]): Remyx API key. If None, will try REMYX_API_KEY env var.
        timeout (int): Code execution timeout in seconds. Default is 300.
        work_dir (Optional[str]): Working directory for code execution.
        auto_remove (bool): Remove container after execution. Default is True.
        stop_container (bool): Stop container after execution. Default is True.
        **kwargs: Additional arguments passed to DockerCommandLineCodeExecutor.

    Raises:
        ImportError: If remyxai package is not installed.
        ValueError: If arxiv_id not found or doesn't have Docker image.
        RuntimeError: If Docker is not available.

    Example:
        >>> from autogen import ConversableAgent
        >>> from autogen.coding import RemyxCodeExecutor
        >>>
        >>> # Create executor for a specific paper
        >>> executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        >>>
        >>> # Create agent with executor
        >>> agent = ConversableAgent(
        ...     "executor",
        ...     llm_config=False,
        ...     code_execution_config={"executor": executor}
        ... )
    """

    SUPPORTED_LANGUAGES: ClassVar[list[str]] = ["python", "bash", "sh"]

    def __init__(
        self,
        arxiv_id: str | None = None,
        image: str | None = None,
        api_key: str | None = None,
        timeout: int = 300,
        work_dir: str | None = None,
        auto_remove: bool = True,
        stop_container: bool = True,
        **kwargs,
    ):
        """Initialize Remyx Code Executor."""
        if RemyxSearchClient is None or remyxai_get_asset is None:
            raise ImportError(
                "Missing dependencies for RemyxCodeExecutor. "
                "Please install with: pip install ag2[remyx]"
            )

        # Load environment variables if dotenv available
        if _load_dotenv is not None:
            _load_dotenv()

        self.arxiv_id = arxiv_id
        self.api_key = api_key or os.getenv("REMYX_API_KEY")
        self._asset_metadata = None

        # Fetch asset metadata if arxiv_id provided
        if arxiv_id and not image:
            # Use remyxai package to fetch metadata
            asset = remyxai_get_asset(arxiv_id)
            
            if not asset:
                raise ValueError(
                    f"Paper {arxiv_id} not found in Remyx catalog. "
                    f"Search for papers with: from remyxai.client.search import SearchClient"
                )

            if not asset.has_docker:
                raise ValueError(
                    f"Paper {arxiv_id} does not have a Docker image. "
                    f"Search for papers with Docker using: has_docker=True filter"
                )

            # Convert Asset to dict for storage
            self._asset_metadata = asset.to_dict()
            image = asset.docker_image
            logger.info(f"Using Docker image for {arxiv_id}: {image}")

        if not image:
            raise ValueError("Either arxiv_id or image must be provided")

        # Prepare container environment from asset metadata
        container_env = {}
        if self._asset_metadata:
            for var in self._asset_metadata.get("environment_vars", []):
                if os.getenv(var):
                    container_env[var] = os.getenv(var)
                else:
                    logger.warning(f"Environment variable {var} not set (may be needed by paper)")

        # Merge with user-provided environment
        container_kwargs = kwargs.get("container_create_kwargs", {})
        if container_env:
            existing_env = container_kwargs.get("environment", {})
            container_env.update(existing_env)
            container_kwargs["environment"] = container_env

        # Set working directory from metadata if available
        if self._asset_metadata and "working_directory" in self._asset_metadata:
            container_kwargs["working_dir"] = self._asset_metadata["working_directory"]

        kwargs["container_create_kwargs"] = container_kwargs

        # Initialize parent DockerCommandLineCodeExecutor
        super().__init__(
            image=image,
            timeout=timeout,
            work_dir=work_dir,
            auto_remove=auto_remove,
            stop_container=stop_container,
            **kwargs,
        )

        logger.info(f"Remyx executor initialized for {arxiv_id or image}")

    @property
    def code_extractor(self) -> CodeExtractor:
        """Export a code extractor that can be used by an agent."""
        return MarkdownCodeExtractor()

    @property
    def paper_info(self) -> dict | None:
        """Get paper metadata if available."""
        return self._asset_metadata

    def get_paper_context(self) -> str:
        """
        Get formatted context about the paper for agent prompts.
        
        This is useful for creating the system message for the Writer Agent
        in the 2-agent interactive pattern.
        """
        if not self._asset_metadata:
            return "No paper metadata available."

        context = f"""Paper Information:
Title: {self._asset_metadata.get('title', 'Unknown')}
arXiv ID: {self._asset_metadata.get('arxiv_id', 'Unknown')}
GitHub: {self._asset_metadata.get('github_url', 'Not available')}
Working Directory: {self._asset_metadata.get('working_directory', '/app')}"""

        if self._asset_metadata.get("reasoning"):
            context += f"\n\nContext:\n{self._asset_metadata['reasoning']}"

        if self._asset_metadata.get("quickstart_hint"):
            context += f"\n\nQuickstart:\n{self._asset_metadata['quickstart_hint']}"

        return context

    def __repr__(self) -> str:
        """String representation."""
        if self.arxiv_id:
            return f"RemyxCodeExecutor(arxiv_id='{self.arxiv_id}')"
        return f"RemyxCodeExecutor(image='{self._image}')"


# ============================================================================
# HELPER FUNCTIONS FOR 2-AGENT INTERACTIVE PATTERN
# ============================================================================
# These functions help users easily create the 2-agent pattern from experiment_ops.py

def create_interactive_agents(
    arxiv_id: str,
    exploration_goal: str | None = None,
    llm_model: str = "gpt-4o-mini",
    timeout: int = 300,
    human_input_mode: str = "ALWAYS",
):
    """
    Create a 2-agent system for interactive research paper exploration.
    
    This replicates the pattern from RepoRanger's experiment_ops.py:
    - Executor Agent: Runs code in Docker (no LLM)
    - Writer Agent: Explores and writes code (has LLM)
    - Human: Guides the exploration (when human_input_mode="ALWAYS")
    
    This is the recommended way to use RemyxCodeExecutor for interactive exploration.
    
    Args:
        arxiv_id: The arXiv paper ID to explore
        exploration_goal: Optional custom exploration goal
        llm_model: LLM model for the writer agent
        timeout: Execution timeout in seconds
        human_input_mode: "ALWAYS" for interactive, "NEVER" for automated
        
    Returns:
        Tuple of (executor_agent, writer_agent, executor_instance)
        
    Example:
        >>> from autogen import LLMConfig
        >>> from autogen.coding import create_interactive_agents
        >>> 
        >>> # Create 2-agent system
        >>> executor_agent, writer_agent, executor = create_interactive_agents(
        ...     arxiv_id="2010.11929v2",
        ...     exploration_goal="Help me understand CLIP's architecture"
        ... )
        >>> 
        >>> # Start interactive exploration
        >>> executor_agent.initiate_chat(
        ...     writer_agent,
        ...     message="Let's explore this paper..."
        ... )
    """
    from autogen import ConversableAgent, LLMConfig
    
    # Create RemyxCodeExecutor
    executor = RemyxCodeExecutor(
        arxiv_id=arxiv_id,
        timeout=timeout,
        environment={
            "HF_TOKEN": os.getenv("HF_TOKEN", ""),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
        }
    )
    
    # Executor Agent - Runs code in Docker (no LLM)
    executor_agent = ConversableAgent(
        "code_executor",
        llm_config=False,
        code_execution_config={"executor": executor},
        human_input_mode="NEVER",
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
    )
    
    # Default exploration goal
    default_goal = """Perform an interactive exploration of this research paper:

**Phase 1: Understanding** (2-3 turns)
1. Examine the directory structure
2. Read README and identify key files
3. Understand the paper's implementation

**Phase 2: Experimentation** (3-5 turns)
4. Run a minimal working example
5. Experiment with different parameters
6. Generate visualizations if applicable

**Phase 3: Analysis** (2-3 turns)
7. Explain key implementation details
8. Answer any questions about the code
9. Suggest possible modifications/experiments

Work step-by-step. Wait for human guidance between phases.
Type TERMINATE when exploration is complete."""

    # Get paper context for system message
    paper_context = executor.get_paper_context()
    
    # Writer Agent - Explores and writes code (has LLM)
    system_message = f"""{paper_context}

**Your Mission:**
{exploration_goal or default_goal}

**Important Guidelines:**
- Repository is at /app with all dependencies installed
- Execute ONE command at a time - don't rush
- Use absolute paths starting with /app
- Be conversational and explain your actions
- If you encounter errors, debug step-by-step
- Wait for human feedback before major actions (if human_input_mode="ALWAYS")
- Focus on lightweight demos, not full training
- You can install additional packages if needed

**What You Can Do:**
✓ Read and analyze code
✓ Execute Python/bash commands  
✓ Modify code for experiments
✓ Generate plots and visualizations
✓ Install additional dependencies
✓ Answer questions about implementation
✓ Suggest improvements or experiments

Begin by exploring the repository structure to understand what's available."""

    writer_agent = ConversableAgent(
        "research_explorer",
        system_message=system_message,
        llm_config=LLMConfig(
            model=llm_model,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        code_execution_config=False,
        max_consecutive_auto_reply=50,
        human_input_mode=human_input_mode,
    )
    
    return executor_agent, writer_agent, executor


def explore_paper(
    arxiv_id: str,
    exploration_goal: str | None = None,
    interactive: bool = True,
):
    """
    Quick helper to launch an interactive exploration session.
    
    This is a convenience function that:
    1. Creates the 2-agent system
    2. Starts the chat
    3. Returns the result
    
    Args:
        arxiv_id: The arXiv paper ID
        exploration_goal: Optional custom goal
        interactive: If True, human guides exploration
        
    Returns:
        Chat result
        
    Example:
        >>> from autogen.coding import explore_paper
        >>> 
        >>> # Launch interactive session
        >>> result = explore_paper(
        ...     arxiv_id="2010.11929v2",
        ...     exploration_goal="Run CLIP demo with custom images"
        ... )
    """
    human_mode = "ALWAYS" if interactive else "NEVER"
    
    executor_agent, writer_agent, _ = create_interactive_agents(
        arxiv_id=arxiv_id,
        exploration_goal=exploration_goal,
        human_input_mode=human_mode,
    )
    
    print("="*80)
    print("🔬 Interactive Research Exploration Session")
    print("="*80)
    print(f"📄 Paper: {arxiv_id}")
    
    if interactive:
        print("\n💬 INTERACTIVE MODE")
        print("   - Press ENTER to continue")
        print("   - Type guidance/questions")
        print("   - Type 'exit' to end")
    else:
        print("\n🤖 AUTOMATED MODE")
    
    print("="*80)
    print()
    
    result = executor_agent.initiate_chat(
        writer_agent,
        message="Let's begin exploring this research paper. Start by examining the directory structure.",
    )
    
    print("\n" + "="*80)
    print("✅ Exploration Complete!")
    print("="*80)
    
    return result
