# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Iterable
from typing import Annotated

from pydantic import Field

from autogen.beta.exceptions import SkillDownloadError, SkillInstallError
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import Toolkit, tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.skills.local_skills import SkillsToolkit
from autogen.beta.tools.skills.runtime import LocalRuntime, SkillRuntime

from .client import SkillsClient
from .config import SkillsClientConfig
from .extractor import format_install_result
from .lock import SkillsLock


class SkillSearchToolkit(SkillsToolkit):
    """Toolkit for dynamically searching and installing skills from the
    `skills.sh <https://skills.sh>`_ ecosystem.

    Does **not** require Node.js. Uses HTTP + GitHub Tarball API directly.
    A ``GITHUB_TOKEN`` environment variable is read automatically to raise the
    GitHub rate limit from 60 to 5,000 requests per hour.

    Example::

        import asyncio
        from autogen.beta import Agent
        from autogen.beta.config import AnthropicConfig
        from autogen.beta.tools import SkillSearchToolkit

        config = AnthropicConfig(model="claude-sonnet-4-5")
        skills = SkillSearchToolkit()

        agent = Agent(
            "coder",
            "You are a helpful coding assistant. Use skills to extend your capabilities.",
            config=config,
            tools=[skills],
        )


        async def main():
            reply = await agent.ask("Find and install a skill for React best practices, then tell me the top 3 rules.")
            print(await reply.content())


        asyncio.run(main())

    Custom configuration::

        from autogen.beta.tools import SkillSearchToolkit, SkillsClientConfig, LocalRuntime

        skills = SkillSearchToolkit(
            runtime=LocalRuntime(
                dir="./my-skills",
                extra_paths=["./extra-skills"],
                cleanup=True,
                timeout=30,
                blocked=["rm -rf"],
            ),
            client=SkillsClientConfig(github_token="ghp_...", proxy="http://proxy:8080"),
        )

    Individual tools are available as methods::

        agent = Agent("a", config=config, tools=[skills.search_skills(), skills.install_skill()])
    """

    __slots__ = ("_client", "_lock")

    def __init__(
        self,
        runtime: SkillRuntime | str | os.PathLike[str] | None = None,
        *,
        client: SkillsClientConfig | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if runtime is not None:
            self._runtime: SkillRuntime = LocalRuntime.ensure_runtime(runtime)
        else:
            self._runtime = LocalRuntime()

        self._client = SkillsClient(client)
        self._lock = SkillsLock(self._runtime.lock_dir / "skills-lock.json")

        Toolkit.__init__(
            self,
            self.list_skills(),
            self.load_skill(),
            self.run_skill_script(),
            self.search_skills(),
            self.install_skill(),
            self.remove_skill(),
            name="local_skills_toolkit",
            middleware=middleware,
        )

    def search_skills(
        self,
        *,
        name: str = "search_skills",
        description: str = "Search for skills on skills.sh. Returns a formatted list of matching skills with ready-to-use install commands.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client = self._client

        @tool(name=name, description=description, middleware=middleware)
        async def _search_skills(
            query: Annotated[
                str,
                Field(description='Search query (e.g. "react performance").'),
            ],
            limit: int = Field(
                default=10,
                description="Maximum number of results to return.",
            ),
        ) -> str:
            try:
                skills = await client.search(query, limit)
            except Exception as e:
                return f"Error searching skills.sh: {e}"

            if not skills:
                return f'No skills found for "{query}".'

            lines: list[str] = [f'Found {len(skills)} skill(s) for "{query}":\n']
            for i, s in enumerate(skills, 1):
                skill_name = s.get("name") or s.get("skillId") or "unknown"
                installs: int = s.get("installs", 0)
                skill_id_val: str = s.get("skillId") or ""
                source: str = s.get("source") or ""
                install_id = f"{source}/{skill_id_val}" if skill_id_val and source else source or skill_id_val
                lines.append(f"{i}. {skill_name} ({installs:,} installs)")
                lines.append(f'   \u2192 install_skill("{install_id}")')
                lines.append("")
            return "\n".join(lines)

        return _search_skills

    def install_skill(
        self,
        *,
        name: str = "install_skill",
        description: str = "Download and install a skill from skills.sh.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        client = self._client
        lock = self._lock
        runtime = self._runtime

        @tool(name=name, description=description, middleware=middleware)
        async def _install_skill(
            skill_id: Annotated[
                str,
                Field(
                    description=(
                        "The skill identifier from search results, e.g.: "
                        '"vercel-labs/agent-skills/react-best-practices" (monorepo), '
                        '"mvanhorn/last30days-skill" (standalone repo).'
                    )
                ),
            ],
        ) -> str:
            parts = skill_id.split("/")
            if len(parts) >= 3:
                source, sid = f"{parts[0]}/{parts[1]}", "/".join(parts[2:])
            elif len(parts) == 2:
                source, sid = skill_id, ""
            else:
                return f"Invalid skill_id format: {skill_id!r}. Expected 'owner/repo/skill-name' or 'owner/repo'."

            try:
                runtime.ensure_storage()
                meta, computed_hash = await client.download_skill(source, sid, runtime)
                lock.record(meta.name, source, computed_hash)
                runtime.invalidate()
                install_dir = runtime.lock_dir
                return format_install_result(meta, install_dir)
            except (SkillDownloadError, SkillInstallError) as e:
                return str(e)
            except Exception as e:
                return f"Error installing skill: {e}"

        return _install_skill

    def remove_skill(
        self,
        *,
        name: str = "remove_skill",
        description: str = "Remove an installed skill by name.",
        middleware: Iterable[ToolMiddleware] = (),
    ) -> FunctionTool:
        lock = self._lock
        runtime = self._runtime

        @tool(name=name, description=description, middleware=middleware)
        def _remove_skill(
            name: Annotated[
                str,
                Field(description="Skill name as returned by list_skills()."),
            ],
        ) -> str:
            try:
                runtime.remove(name)
            except (ValueError, FileNotFoundError) as e:
                return str(e)
            lock.remove(name)
            runtime.invalidate()
            return f"Removed: {name}"

        return _remove_skill
