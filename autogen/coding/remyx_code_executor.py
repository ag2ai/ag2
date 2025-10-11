# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Remyx code executor implementation for research paper execution."""

import logging
import os
from collections.abc import Callable
from hashlib import md5
from typing import ClassVar

from pydantic import Field

from ..code_utils import TIMEOUT_MSG, _cmd
from ..doc_utils import export_module
from .base import CodeBlock, CodeExecutor, CodeExtractor, CodeResult, CommandLineCodeResult
from .docker_commandline_code_executor import DockerCommandLineCodeExecutor
from .markdown_code_extractor import MarkdownCodeExtractor
from .utils import _get_file_name_from_content, silence_pip

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
    4. Enable interactive exploration with AI agents

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
        Basic execution:
        >>> from autogen import ConversableAgent
        >>> from autogen.coding import RemyxCodeExecutor
        >>>
        >>> # Create executor for a paper
        >>> executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        >>>
        >>> # Create agent with executor
        >>> agent = ConversableAgent(
        ...     "executor",
        ...     llm_config=False,
        ...     code_execution_config={"executor": executor}
        ... )
        
        Interactive exploration (recommended):
        >>> executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
        >>> result = executor.explore(
        ...     goal="Help me understand the main innovation from this paper",
        ...     interactive=True
        ... )
    """

    SUPPORTED_LANGUAGES: ClassVar[list[str]] = ["python", "bash", "sh"]
    
    # Language to file extension mapping (parent class uses lang name directly as extension)
    _LANG_EXT_MAP: ClassVar[dict[str, str]] = {
        "python": "py",
        "bash": "sh",
        "sh": "sh",
        "shell": "sh",
    }

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

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> CommandLineCodeResult:
        """Execute code blocks with correct file extensions.
        
        Overrides parent to fix file extension issue where 'python' becomes '.python'
        instead of '.py'.
        """
        if len(code_blocks) == 0:
            raise ValueError("No code blocks to execute.")

        outputs = []
        files = []
        last_exit_code = 0
        
        for code_block in code_blocks:
            lang = self.LANGUAGE_ALIASES.get(code_block.language.lower(), code_block.language.lower())
            if lang not in self.DEFAULT_EXECUTION_POLICY:
                outputs.append(f"Unsupported language {lang}\n")
                last_exit_code = 1
                break

            execute_code = self.execution_policies.get(lang, False)
            code = silence_pip(code_block.code, lang)

            # Check if there is a filename comment
            try:
                filename = _get_file_name_from_content(code, self._work_dir)
            except ValueError:
                outputs.append("Filename is not in the workspace")
                last_exit_code = 1
                break

            if not filename:
                # FIX: Use correct file extension mapping
                ext = self._LANG_EXT_MAP.get(lang, lang)
                filename = f"tmp_code_{md5(code.encode()).hexdigest()}.{ext}"

            code_path = self._work_dir / filename
            with code_path.open("w", encoding="utf-8") as fout:
                fout.write(code)
            files.append(code_path)

            if not execute_code:
                outputs.append(f"Code saved to {code_path!s}\n")
                continue

            command = ["timeout", str(self._timeout), _cmd(lang), filename]
            result = self._container.exec_run(command)
            exit_code = result.exit_code
            output = result.output.decode("utf-8")
            if exit_code == 124:
                output += "\n" + TIMEOUT_MSG
            outputs.append(output)

            last_exit_code = exit_code
            if exit_code != 0:
                break

        code_file = str(files[0]) if files else None
        return CommandLineCodeResult(exit_code=last_exit_code, output="".join(outputs), code_file=code_file)

    def get_paper_context(self) -> str:
        """
        Get formatted context about the paper for agent prompts.
        
        This is useful for creating system messages for exploration agents.
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

    def explore(
        self,
        goal: str | None = None,
        interactive: bool = True,
        llm_model: str = "gpt-4o-mini",
        max_turns: int | None = None,
        verbose: bool = True,
    ):
        """
        Explore this research paper interactively with AI agents.
        
        This is the recommended way to understand and experiment with research code.
        Creates a 2-agent system where one agent proposes experiments and another
        executes them in the paper's Docker environment.
        
        Args:
            goal: Optional custom exploration goal. If None, uses a default exploration plan.
            interactive: If True, pauses for human guidance at each step. If False, runs automatically.
            llm_model: The LLM model to use for the exploring agent. Default is "gpt-4o-mini".
            max_turns: Maximum number of conversation turns. If None, continues until termination.
            verbose: If True, prints session header and summary. If False, runs quietly.
        
        Returns:
            The chat result from the exploration session.
        
        Example:
            >>> # Interactive exploration (recommended for learning)
            >>> executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
            >>> result = executor.explore(
            ...     goal="Help me understand the main innovation from this paper",
            ...     interactive=True
            ... )
            
            >>> # Automated exploration (good for batch experiments)
            >>> result = executor.explore(
            ...     goal="Run all examples and benchmarks",
            ...     interactive=False,
            ...     verbose=False  # Quiet mode
            ... )
        """
        from autogen import ConversableAgent, LLMConfig
        
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

        # Get paper context
        paper_context = self.get_paper_context()
        
        # Create system message for writer agent
        system_message = f"""{paper_context}

**Your Mission:**
{goal or default_goal}

**Important Guidelines:**
- Repository is at /app with all dependencies installed
- Execute ONE command at a time - don't rush
- Use absolute paths starting with /app
- Be conversational and explain your actions
- If you encounter errors, debug step-by-step
- Wait for human feedback before major actions (if interactive mode)
- Focus on lightweight demos unless instructed otherwise
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

        # Create executor agent (no LLM)
        executor_agent = ConversableAgent(
            "code_executor",
            llm_config=False,
            code_execution_config={"executor": self},
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        )
        
        # Create writer agent (has LLM)
        writer_agent = ConversableAgent(
            "research_explorer",
            system_message=system_message,
            llm_config=LLMConfig(
                model=llm_model,
                api_key=os.getenv("OPENAI_API_KEY"),
            ),
            code_execution_config=False,
            max_consecutive_auto_reply=50,
            human_input_mode="ALWAYS" if interactive else "NEVER",
        )
        
        # Print session header
        if verbose:
            print("=" * 80)
            print("🔬 Interactive Research Exploration Session")
            print("=" * 80)
            print(f"📄 Paper: {self.arxiv_id or 'Custom image'}")
            
            if interactive:
                print("\n💬 INTERACTIVE MODE")
                print("   - Press ENTER to continue")
                print("   - Type guidance/questions")
                print("   - Type 'exit' to end")
            else:
                print("\n🤖 AUTOMATED MODE")
            
            print("=" * 80)
            print()
        
        # Start exploration
        result = executor_agent.initiate_chat(
            writer_agent,
            message="Let's begin exploring this research paper. Start by examining the directory structure.",
            max_turns=max_turns,
        )
        
        # Print summary
        if verbose:
            print("\n" + "=" * 80)
            print("✅ Exploration Complete!")
            print("=" * 80)
            print(f"\n📊 Session Summary:")
            print(f"   • Total messages: {len(result.chat_history)}")
            print(f"   • Cost: ${result.cost['usage_including_cached_inference']['total_cost']:.4f}")
            
            if result.summary:
                print(f"\n💬 Final Status:")
                # Print first 200 chars of summary
                summary_preview = result.summary[:200] + "..." if len(result.summary) > 200 else result.summary
                print(f"   {summary_preview}")
            
            print(f"\n💾 Full chat history available in returned object")
            print(f"   Access with: result.chat_history")
            print("=" * 80)
        
        return result

    def create_agents(
        self,
        goal: str | None = None,
        llm_model: str = "gpt-4o-mini",
        human_input_mode: str = "ALWAYS",
    ):
        """
        Create the 2-agent system without starting exploration.
        
        Use this if you want more control over the exploration process.
        Most users should use the simpler `explore()` method instead.
        
        Args:
            goal: Optional custom exploration goal.
            llm_model: The LLM model to use.
            human_input_mode: "ALWAYS" for interactive, "NEVER" for automated.
        
        Returns:
            Tuple of (executor_agent, writer_agent)
        
        Example:
            >>> executor = RemyxCodeExecutor(arxiv_id="2010.11929v2")
            >>> executor_agent, writer_agent = executor.create_agents()
            >>> # Customize the chat further
            >>> result = executor_agent.initiate_chat(
            ...     writer_agent,
            ...     message="Custom starting message",
            ...     max_turns=10
            ... )
        """
        from autogen import ConversableAgent, LLMConfig
        
        # Default goal
        default_goal = """Explore this research paper interactively.
Work step-by-step, explain your actions, and wait for guidance."""

        # Get paper context
        paper_context = self.get_paper_context()
        
        # Create system message
        system_message = f"""{paper_context}

**Your Mission:**
{goal or default_goal}

**Guidelines:**
- Repository is at /app with all dependencies installed
- Execute ONE command at a time
- Use absolute paths starting with /app
- Be conversational and explain your actions
- Debug step-by-step if errors occur

Begin by exploring the repository structure."""

        # Create executor agent
        executor_agent = ConversableAgent(
            "code_executor",
            llm_config=False,
            code_execution_config={"executor": self},
            human_input_mode="NEVER",
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        )
        
        # Create writer agent
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
        
        return executor_agent, writer_agent

    def __repr__(self) -> str:
        """String representation."""
        if self.arxiv_id:
            return f"RemyxCodeExecutor(arxiv_id='{self.arxiv_id}')"
        return f"RemyxCodeExecutor(image='{self._image}')"

    @staticmethod
    def format_chat_result(result) -> str:
        """
        Format a ChatResult object into a readable summary.
        
        Useful for displaying exploration results later.
        
        Args:
            result: The ChatResult object from explore() or initiate_chat()
        
        Returns:
            Formatted string summary
        
        Example:
            >>> result = executor.explore(verbose=False)
            >>> print(RemyxCodeExecutor.format_chat_result(result))
        """
        lines = []
        lines.append("=" * 80)
        lines.append("📊 Exploration Session Summary")
        lines.append("=" * 80)
        
        # Basic stats
        lines.append(f"\n• Total messages: {len(result.chat_history)}")
        lines.append(f"• Chat ID: {result.chat_id}")
        
        # Cost info
        if hasattr(result, 'cost') and result.cost:
            total_cost = result.cost.get('usage_including_cached_inference', {}).get('total_cost', 0)
            lines.append(f"• Cost: ${total_cost:.4f}")
        
        # Summary
        if result.summary:
            lines.append(f"\n💬 Final Status:")
            summary_preview = result.summary[:300] + "..." if len(result.summary) > 300 else result.summary
            for line in summary_preview.split('\n'):
                lines.append(f"   {line}")
        
        # Last few messages
        lines.append(f"\n📝 Last 3 messages:")
        for msg in result.chat_history[-3:]:
            role = msg.get('name', msg.get('role', 'unknown'))
            content_preview = msg.get('content', '')[:100]
            if len(msg.get('content', '')) > 100:
                content_preview += "..."
            lines.append(f"   [{role}]: {content_preview}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
