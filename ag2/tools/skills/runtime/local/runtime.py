# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import os
import shlex
import shutil
import stat
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ag2.tools.sandbox import Sandbox, SandboxFactory
from ag2.tools.sandbox.adapter import ShellAdapter
from ag2.tools.sandbox.local import LocalSandbox
from ag2.tools.skills.skill_types import Skill

from ..protocol import SkillRuntime
from .loader import SkillLoader, strip_frontmatter

if TYPE_CHECKING:
    from ag2.context import ConversationContext

# Max characters returned from a single resource read before truncation.
_RESOURCE_READ_CAP = 100_000


@dataclass
class LocalRuntime(SkillRuntime):
    """Local filesystem storage and subprocess execution.

    Args:
        dir:         Directory where skills are installed.
                     ``None`` → ``.agents/skills`` (default).
        cleanup:     If ``True``, the install directory is deleted at process exit.
        timeout:     Per-command timeout in seconds. Defaults to 60.
        max_output:  Maximum characters returned from a script run. Defaults to 100,000.
        blocked:     Command prefixes that are not allowed to run. Empty list → nothing blocked.
                     Best-effort only (matches the head command prefix; chaining such
                     as ``;`` / ``|`` / ``&&`` / ``$(...)`` bypasses it) — not a security
                     boundary.
        extra_paths: Additional read-only directories to scan for skills.
                     Installed skills always go to *dir*; these paths are only
                     used for discovery.
        sandbox:     Execution backend for ``run_skill_script``. ``None`` → a local
                     subprocess rooted at the skill's ``scripts/`` directory. Pass a
                     :class:`~ag2.tools.sandbox.Sandbox` /
                     :class:`~ag2.tools.sandbox.SandboxFactory`
                     (e.g. ``DockerEnvironment``) to run scripts inside that backend;
                     the caller is responsible for making the scripts reachable there.

    Example::

        # Default — installs to .agents/skills
        LocalRuntime()

        # Custom install directory
        LocalRuntime("./my-skills")

        # Full control
        LocalRuntime("./my-skills", timeout=30, cleanup=True, blocked=["rm -rf"])

        # Extra read-only search paths
        LocalRuntime("./my-skills", extra_paths=["./shared-skills"])
    """

    dir: str | os.PathLike[str] | None = None
    cleanup: bool = False
    timeout: float = 60
    max_output: int = 100_000
    blocked: list[str] = field(default_factory=list)
    extra_paths: Sequence[str | os.PathLike[str]] | None = None
    sandbox: "Sandbox | SandboxFactory | None" = None

    def __post_init__(self) -> None:
        self._install_dir = Path(self.dir) if self.dir is not None else Path(".agents/skills")
        self._extra: list[Path] = [Path(p) for p in self.extra_paths] if self.extra_paths else []
        # Lenient discovery: a single non-compliant skill (e.g. authored for
        # another client) is skipped with a warning instead of aborting the whole
        # scan. Strict validation still gates the *install* path, where the
        # extractor calls ``SkillLoader.validate_skill_metadata`` directly.
        self._loader = SkillLoader(self._install_dir, *self._extra, strict=False)
        if self.cleanup:
            atexit.register(shutil.rmtree, str(self._install_dir), True)

    @property
    def install_dir(self) -> Path:
        """Resolved install directory."""
        return self._install_dir

    @property
    def lock_dir(self) -> Path:
        return self._install_dir

    @property
    def skills(self) -> list[Skill]:
        return self._loader.discover()

    def read(self, name: str) -> str:
        """Return the model-ready content for *name* (wrapped SKILL.md body).

        Raises ``SkillNotFoundError`` (via the loader) when *name* is unknown.
        """
        skill = self._loader.get_skill(name)
        skill_dir = self._loader.get_path(name)
        # Body only: the frontmatter is already surfaced via the catalog.
        body = strip_frontmatter(self._loader.load(name))
        return _wrap_skill_content(name, body, skill_dir, skill)

    async def read_resource(self, name: str, resource: str, context: "ConversationContext") -> str:
        """Read a bundled resource by its descriptor name (list-based lookup).

        *context* is part of the runtime protocol (used by callable-backed runtimes
        for dependency injection); a filesystem read ignores it.
        """
        skill = self._loader.get_skill(name)
        skill_dir = self._loader.get_path(name)
        resolved = _resolve_within(skill_dir / resource, skill_dir)
        if resource not in {r.name for r in skill.resources} or resolved is None:
            raise FileNotFoundError(f"resource {resource!r} not found in {skill_dir}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if len(text) > _RESOURCE_READ_CAP:
            return text[:_RESOURCE_READ_CAP] + "\n<!-- resource truncated -->"
        return text

    async def execute(
        self,
        name: str,
        script: str,
        context: "ConversationContext",
        args: dict[str, Any] | Sequence[str] | None = None,
    ) -> str:
        """Run a script of *name* and return its output (list-based lookup).

        *context* is part of the runtime protocol; a subprocess script ignores it.
        """
        if isinstance(args, dict):
            raise TypeError(
                f"file-based script {script!r} requires positional string arguments (an array); "
                "named arguments (an object) are only supported for in-process scripts"
            )
        skill = self._loader.get_skill(name)
        scripts_dir = self._loader.get_path(name) / "scripts"
        resolved_script = _resolve_within(scripts_dir / script, scripts_dir)
        if script not in {s.name for s in skill.scripts} or resolved_script is None:
            raise FileNotFoundError(f"script {script!r} not found in {scripts_dir}")
        command = _script_command(resolved_script)
        if args:
            command.extend(args)
        # async + await env.run(...) so the command runs in the agent's own
        # event loop. A sync path would drive remote backends via a throwaway
        # asyncio.run() per call (a fresh loop each time), breaking clients
        # bound to the first loop (e.g. Daytona's httpx keep-alive pool).
        env = self.shell(scripts_dir)
        return await env.run(shlex.join(command))

    def load(self, name: str) -> str:
        return self._loader.load(name)

    def get_path(self, name: str) -> Path:
        return self._loader.get_path(name)

    def invalidate(self) -> None:
        self._loader.invalidate()

    @classmethod
    def ensure_runtime(cls, runtime: SkillRuntime | str | os.PathLike[str]) -> SkillRuntime:
        # Test for a path, not ``isinstance(runtime, SkillRuntime)``: SkillRuntime
        # is a ``runtime_checkable`` Protocol, and on Python <=3.11 that isinstance
        # check calls every member — including the ``lock_dir`` property, whose
        # getter raises on read-only runtimes like ``MemoryRuntime``. Anything that
        # is not a path is already a runtime.
        if isinstance(runtime, (str, os.PathLike)):
            return cls(dir=runtime)
        return runtime

    def ensure_storage(self) -> None:
        self._install_dir.mkdir(parents=True, exist_ok=True)

    def install(self, source: Path, name: str) -> None:
        dest = self._install_dir / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)

    def remove(self, name: str) -> None:
        target = (self._install_dir / name).resolve()
        if not target.is_relative_to(self._install_dir.resolve()):
            raise ValueError(f"Cannot remove '{name}': path traversal detected")
        if not target.exists():
            raise FileNotFoundError(f"Cannot remove '{name}': skill not found in {self._install_dir}")
        shutil.rmtree(target)

    def shell(self, scripts_dir: Path) -> ShellAdapter:
        if self.sandbox is not None:
            # A user-supplied backend (Docker/Daytona/…). The backend owns its
            # own workdir; the caller is responsible for making scripts
            # reachable inside it (e.g. a bind-mounted host_path).
            return ShellAdapter(
                self.sandbox,
                blocked=self.blocked or None,
                timeout=self.timeout,
            )
        sandbox = LocalSandbox(
            path=scripts_dir,
            cleanup=False,
            timeout=self.timeout,
            max_output=self.max_output,
        )
        return ShellAdapter(
            sandbox,
            blocked=self.blocked or None,
            timeout=self.timeout,
        )


def _resolve_within(path: Path, base: Path) -> Path | None:
    """Resolve *path* (following symlinks) and return it only when it is a file
    that stays inside *base*; otherwise return ``None``.
    """
    resolved = path.resolve()
    if not resolved.is_file() or not resolved.is_relative_to(base.resolve()):
        return None
    return resolved


def _script_command(resolved_script: Path) -> list[str]:
    """Build the argv to run *resolved_script* from its own directory.

    A shebang takes precedence; otherwise ``.py``/``.sh`` map to their
    interpreters, and anything else is made executable and run directly.
    """
    first_line = resolved_script.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
    if first_line.startswith("#!"):
        resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return [f"./{resolved_script.name}"]
    if resolved_script.suffix.lower() == ".py":
        return ["python3", f"./{resolved_script.name}"]
    if resolved_script.suffix.lower() == ".sh":
        return ["sh", f"./{resolved_script.name}"]
    resolved_script.chmod(resolved_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return [f"./{resolved_script.name}"]


def _wrap_skill_content(name: str, body: str, skill_dir: Path, skill: Skill) -> str:
    """Wrap a SKILL.md body with identifying tags, base path, and resource list."""
    lines = [
        f'<skill_content name="{name}">',
        body.strip(),
        "",
        f"Skill directory: {skill_dir}",
        "Relative paths in this skill are relative to the skill directory.",
    ]
    if skill.resources:
        lines.append("<skill_resources>")
        lines.extend(f"  <file>{r.name}</file>" for r in skill.resources)
        if skill.resources_truncated:
            lines.append("  <!-- resource list truncated -->")
        lines.append("</skill_resources>")
    lines.append("</skill_content>")
    return "\n".join(lines)
