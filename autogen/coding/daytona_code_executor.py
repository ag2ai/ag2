# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
"""Daytona code executor implementation."""

from __future__ import annotations

import atexit
import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

logger = logging.getLogger(__name__)

from pydantic import Field

from ..doc_utils import export_module
from .base import CodeBlock, CodeExtractor, CodeResult
from .markdown_code_extractor import MarkdownCodeExtractor

if TYPE_CHECKING:
    from daytona import Image

try:
    from daytona import (
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        Daytona,
        DaytonaConfig,
        DaytonaError,
        DaytonaNotFoundError,
        DaytonaRateLimitError,
        DaytonaTimeoutError,
        Resources,
    )
except ImportError:
    Daytona = None  # type: ignore[assignment,misc]
    DaytonaConfig = None  # type: ignore[assignment,misc]
    CreateSandboxFromSnapshotParams = None  # type: ignore[assignment,misc]
    CreateSandboxFromImageParams = None  # type: ignore[assignment,misc]
    Resources = None  # type: ignore[assignment,misc]
    DaytonaError = None  # type: ignore[assignment,misc]
    DaytonaNotFoundError = None  # type: ignore[assignment,misc]
    DaytonaRateLimitError = None  # type: ignore[assignment,misc]
    DaytonaTimeoutError = None  # type: ignore[assignment,misc]


@export_module("autogen.coding")
@dataclass
class DaytonaSandboxResources:
    """Resource limits for a Daytona sandbox.

    All fields are optional — only set the ones you want to constrain.
    Only applied when using a custom image (the ``image`` parameter);
    ignored when a snapshot is specified.

    Args:
        cpu: Number of CPU cores to allocate.
        memory: Memory in gigabytes.
        disk: Disk space in gigabytes.
    """

    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None


@export_module("autogen.coding")
class DaytonaCodeResult(CodeResult):
    """A code result class for Daytona executor."""

    sandbox_id: str | None = Field(default=None, description="The Daytona sandbox ID for this result.")


@export_module("autogen.coding")
class DaytonaCodeExecutor:
    """A code executor that runs code blocks inside a Daytona sandbox.

    Creates a single sandbox on initialization and reuses it across all
    execute_code_blocks() calls. Auto-stop is disabled so the sandbox stays
    alive for the duration of the session. The sandbox is deleted when delete()
    is called, on process exit (via atexit), or when used as a context manager.

    Python blocks are executed via process.code_run(). All other supported
    languages (bash, sh, javascript, typescript) are written to a temp file
    in the sandbox and executed with the appropriate runtime binary.

    Args:
        api_key: Daytona API key. If None, reads from DAYTONA_API_KEY env var.
        api_url: Daytona API URL. If None, reads from DAYTONA_API_URL env var.
        target: Target region (e.g. "us", "eu"). If None, reads from DAYTONA_TARGET env var.
        timeout: Per-execution timeout in seconds. Default is 60.
        snapshot: Sandbox snapshot name to use. Takes priority over ``image``.
            Uses Daytona default if neither snapshot nor image is provided.
        image: Custom Docker image for the sandbox. Accepts an image name string
            (e.g. ``"python:3.12"``) or a Daytona ``Image`` object for a
            declarative build. Ignored when ``snapshot`` is also set.
        name: Human-readable name for the sandbox. Auto-generated if None.
        env_vars: Environment variables to set inside the sandbox. Useful for
            passing secrets (API keys, database URLs, etc.) to the running code.
        resources: CPU/memory/disk limits for the sandbox. Only applied when
            using a custom image; ignored for snapshot-based sandboxes.

    Raises:
        ImportError: If the daytona package is not installed.
        ValueError: If timeout < 1 or both snapshot and image are provided.
        RuntimeError: If sandbox creation fails.

    Example:
        Basic usage:

        ```python
        executor = DaytonaCodeExecutor(api_key="...", timeout=120)
        agent = ConversableAgent("coder", code_execution_config={"executor": executor})
        ```

        As a context manager — sandbox is deleted when the with-block exits:

        ```python
        with DaytonaCodeExecutor(api_key="...") as executor:
            agent = ConversableAgent("coder", code_execution_config={"executor": executor})
        ```
    """

    SUPPORTED_LANGUAGES: ClassVar[list[str]] = [
        "python",
        "bash",
        "sh",
        "javascript",
        "typescript",
    ]

    # Maps normalized language name -> file extension for script files
    _LANG_EXT: ClassVar[dict[str, str]] = {
        "bash": "sh",
        "sh": "sh",
        "javascript": "js",
        "typescript": "ts",
    }

    # Maps normalized language name -> command prefix used to run the script file.
    # Python is handled separately via process.code_run() and is not listed here.
    # bash/sh use the file approach (rather than passing code directly to process.exec)
    # to reliably handle multi-line scripts with loops, conditionals, etc.
    # TypeScript flags: skip type-checking and tsconfig discovery, force CommonJS
    # to avoid "Cannot use import statement outside a module" errors in sandboxes
    # that have no tsconfig.json.
    _LANG_CMD: ClassVar[dict[str, str]] = {
        "bash": "bash",
        "sh": "sh",
        "javascript": "node",
        "typescript": (
            "ts-node --transpile-only --skipProject "
            '--compilerOptions \'{"module":"commonjs","moduleResolution":"node"}\''
        ),
    }

    # Aliases used in markdown code fences -> normalized canonical name
    _LANG_ALIASES: ClassVar[dict[str, str]] = {
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "shell": "bash",
    }

    def __init__(  # type: ignore[no-any-unimported]
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        timeout: int = 60,
        snapshot: str | None = None,
        image: str | Image | None = None,
        name: str | None = None,
        env_vars: dict[str, str] | None = None,
        resources: DaytonaSandboxResources | None = None,
    ):
        if Daytona is None:
            raise ImportError(
                "Missing dependencies for DaytonaCodeExecutor. Please install with: pip install ag2[daytona]"
            )

        if timeout < 1:
            raise ValueError("Timeout must be greater than or equal to 1.")

        if snapshot is not None and image is not None:
            raise ValueError(
                "Cannot specify both 'snapshot' and 'image'. "
                "Use 'snapshot' for a named Daytona snapshot, or 'image' for a custom Docker image."
            )

        self._timeout = timeout
        self._snapshot = snapshot
        self._image = image
        self._name = name or f"ag2-{uuid.uuid4().hex[:8]}"
        self._env_vars = env_vars or {}
        self._resources = resources

        # Build config only from explicitly provided values; the SDK reads
        # DAYTONA_API_KEY / DAYTONA_API_URL / DAYTONA_TARGET from the environment
        # for any field that is not explicitly set.
        config_kwargs: dict[str, str] = {}
        if api_key is not None:
            config_kwargs["api_key"] = api_key
        if api_url is not None:
            config_kwargs["api_url"] = api_url
        if target is not None:
            config_kwargs["target"] = target

        self._client = Daytona(DaytonaConfig(**config_kwargs))
        self._sandbox = self._create_sandbox()

        # Ensure cleanup even if the user never calls delete() or uses a context manager.
        atexit.register(self.delete)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_sandbox(self) -> Any:
        """Create and return a new Daytona sandbox with auto-stop disabled."""
        try:
            params: CreateSandboxFromSnapshotParams | CreateSandboxFromImageParams  # type: ignore[no-any-unimported]
            if self._snapshot is not None:
                # Explicit snapshot always wins — resources don't apply here.
                params = CreateSandboxFromSnapshotParams(
                    snapshot=self._snapshot,
                    env_vars=self._env_vars,
                    name=self._name,
                    auto_stop_interval=0,
                )
            elif self._image is not None:
                # Custom Docker image — forward resources if any field is set.
                sdk_resources = None
                r = self._resources
                if r is not None and any(v is not None for v in (r.cpu, r.memory, r.disk)):
                    sdk_resources = Resources(cpu=r.cpu, memory=r.memory, disk=r.disk)
                params = CreateSandboxFromImageParams(
                    image=self._image,
                    env_vars=self._env_vars,
                    name=self._name,
                    resources=sdk_resources,
                    auto_stop_interval=0,
                )
            else:
                # Neither specified — Daytona uses its default snapshot.
                params = CreateSandboxFromSnapshotParams(
                    env_vars=self._env_vars,
                    name=self._name,
                    auto_stop_interval=0,
                )
            sandbox = self._client.create(params)
            logger.info("Daytona sandbox created (id=%s, name=%s)", sandbox.id, self._name)
            return sandbox
        except DaytonaTimeoutError as e:
            logger.error("Timed out waiting for Daytona sandbox to start: %s", e)
            raise RuntimeError(f"Daytona sandbox creation timed out: {e}") from e
        except DaytonaRateLimitError as e:
            logger.error("Daytona rate limit hit during sandbox creation: %s", e)
            raise RuntimeError(f"Daytona rate limit exceeded during sandbox creation: {e}") from e
        except DaytonaError as e:
            logger.error("Daytona error during sandbox creation: %s", e)
            raise RuntimeError(f"Failed to create Daytona sandbox: {e}") from e
        except Exception as e:
            logger.error("Unexpected error during sandbox creation: %s", e)
            raise RuntimeError(f"Failed to create Daytona sandbox: {e}") from e

    def _normalize_language(self, language: str) -> str:
        """Lowercase and resolve fence aliases to canonical language names."""
        lang = language.lower().strip()
        return self._LANG_ALIASES.get(lang, lang)

    # ------------------------------------------------------------------
    # CodeExecutor protocol
    # ------------------------------------------------------------------

    @property
    def code_extractor(self) -> CodeExtractor:
        """The code extractor used by this executor."""
        return MarkdownCodeExtractor()

    @property
    def timeout(self) -> int:
        """Per-execution timeout in seconds."""
        return self._timeout

    def execute_code_blocks(self, code_blocks: list[CodeBlock]) -> DaytonaCodeResult:
        """Execute code blocks sequentially inside the Daytona sandbox.

        Python blocks are executed via process.code_run(). All other languages
        are written to a temp file and run with the appropriate runtime binary
        via process.exec().

        Execution stops on the first failure and returns that block's result.
        On full success, outputs from all blocks are joined with newlines.

        Args:
            code_blocks: The code blocks to execute.

        Returns:
            DaytonaCodeResult with exit_code, combined output, and sandbox_id.
        """
        if not code_blocks:
            return DaytonaCodeResult(exit_code=0, output="", sandbox_id=self._sandbox.id)

        outputs: list[str] = []

        for code_block in code_blocks:
            lang = self._normalize_language(code_block.language)

            if lang not in self.SUPPORTED_LANGUAGES:
                return DaytonaCodeResult(
                    exit_code=1,
                    output=(
                        f"Unsupported language: {code_block.language!r}. "
                        f"Supported languages: {', '.join(self.SUPPORTED_LANGUAGES)}"
                    ),
                    sandbox_id=self._sandbox.id,
                )

            logger.info("Executing %s code block in sandbox %s", lang, self._sandbox.id)
            try:
                if lang == "python":
                    response = self._sandbox.process.code_run(code_block.code, timeout=self._timeout)
                else:
                    ext = self._LANG_EXT[lang]
                    script_path = f"/tmp/ag2_{uuid.uuid4().hex}.{ext}"

                    self._sandbox.fs.upload_file(code_block.code.encode("utf-8"), script_path)

                    cmd = f"{self._LANG_CMD[lang]} {script_path}"
                    response = self._sandbox.process.exec(cmd, timeout=self._timeout)

                    # Best-effort cleanup — don't let a stale file mask the real result.
                    try:
                        self._sandbox.fs.delete_file(script_path)
                    except DaytonaNotFoundError:
                        pass  # File already gone — expected
                    except Exception as e:
                        logger.debug("Failed to delete temp script %s: %s", script_path, e)

            except DaytonaTimeoutError as e:
                logger.warning("Execution timed out for %s code block in sandbox %s: %s", lang, self._sandbox.id, e)
                return DaytonaCodeResult(
                    exit_code=1,
                    output=f"Execution timed out: {e}",
                    sandbox_id=self._sandbox.id,
                )
            except DaytonaRateLimitError as e:
                logger.warning("Rate limit hit executing %s code block in sandbox %s: %s", lang, self._sandbox.id, e)
                return DaytonaCodeResult(
                    exit_code=1,
                    output=f"Daytona rate limit exceeded: {e}",
                    sandbox_id=self._sandbox.id,
                )
            except DaytonaError as e:
                logger.warning("Daytona error executing %s code block in sandbox %s: %s", lang, self._sandbox.id, e)
                return DaytonaCodeResult(
                    exit_code=1,
                    output=f"Daytona error: {e}",
                    sandbox_id=self._sandbox.id,
                )
            except Exception as e:
                logger.warning("Unexpected error executing %s code block in sandbox %s: %s", lang, self._sandbox.id, e)
                return DaytonaCodeResult(
                    exit_code=1,
                    output=f"Error executing code block: {e}",
                    sandbox_id=self._sandbox.id,
                )
            if response.exit_code != 0:
                return DaytonaCodeResult(
                    exit_code=response.exit_code,
                    output=response.result,
                    sandbox_id=self._sandbox.id,
                )

            outputs.append(response.result)

        return DaytonaCodeResult(
            exit_code=0,
            output="\n".join(outputs),
            sandbox_id=self._sandbox.id,
        )

    def restart(self) -> None:
        """Reset the executor by deleting the current sandbox and creating a fresh one.

        Called by agent.reset() when starting a new conversation. Clears all
        accumulated state — created files, installed packages, process state.
        """
        logger.info("Restarting Daytona executor — deleting sandbox %s", self._sandbox.id if self._sandbox else "None")
        try:
            self._sandbox.delete()
        except DaytonaNotFoundError:
            pass  # Already deleted — expected
        except Exception as e:
            logger.debug("Suppressed exception during sandbox deletion on restart: %s", e)
        self._sandbox = self._create_sandbox()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def delete(self) -> None:
        """Delete the sandbox and release all associated resources.

        Safe to call multiple times. Called automatically on process exit via atexit.
        Unregisters the atexit handler to prevent double-execution and handler accumulation.
        """
        atexit.unregister(self.delete)
        if self._sandbox is not None:
            logger.info("Deleting Daytona sandbox %s", self._sandbox.id)
            try:
                self._sandbox.delete()
            except DaytonaNotFoundError:
                pass  # Already deleted — expected
            except Exception as e:
                logger.debug("Suppressed exception during sandbox deletion: %s", e)
            self._sandbox = None  # type: ignore[assignment]

    def __enter__(self) -> DaytonaCodeExecutor:
        return self

    def __exit__(self, *args: object) -> None:
        self.delete()
