# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

from autogen.beta.exceptions import SkillDownloadError, SkillInstallError
from autogen.beta.middleware import ToolMiddleware
from autogen.beta.tools.final import tool
from autogen.beta.tools.final.function_tool import FunctionTool
from autogen.beta.tools.toolkits.skills.local_skills import SkillsToolkit
from autogen.beta.tools.toolkits.skills.runtime import LocalRuntime, SkillRuntime

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
            LocalRuntime(
                dir="./my-skills",
                extra_paths=["./extra-skills"],
                cleanup=True,
                timeout=30,
                blocked=["rm -rf"],
            ),
            client=SkillsClientConfig(github_token="ghp_...", proxy="http://proxy:8080"),
        )

    Individual tools are available as attributes::

        agent = Agent("a", config=config, tools=[skills.search_skills, skills.install_skill])
    """

    search_skills: FunctionTool
    install_skill: FunctionTool
    remove_skill: FunctionTool
    list_skills: FunctionTool
    load_skill: FunctionTool
    run_skill_script: FunctionTool

    def __init__(
        self,
        runtime: SkillRuntime | None = None,
        *,
        client: SkillsClientConfig | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        _runtime: SkillRuntime = runtime if runtime is not None else LocalRuntime()

        _client = SkillsClient(client)
        lock = SkillsLock(_runtime.lock_dir / "skills-lock.json")

        super().__init__(runtime)

        self.search_skills = _make_search_tool(_client)
        self.install_skill = _make_install_tool(_client, lock, _runtime)
        self.remove_skill = _make_remove_tool(_runtime, lock)

        super(SkillsToolkit, self).__init__(
            self.search_skills,
            self.install_skill,
            self.remove_skill,
            *self.tools,
            name="local_skills_toolkit",
            middleware=middleware,
        )


def _make_search_tool(client: SkillsClient) -> FunctionTool:
    @tool
    async def search_skills(query: str, limit: int = 10) -> str:
        """Search for skills on skills.sh.

        Returns a formatted list of matching skills with ready-to-use install commands.

        Args:
            query: Search query (e.g. ``"react performance"``).
            limit: Maximum number of results to return (default: 10).
        """
        try:
            skills = await client.search(query, limit)
        except Exception as e:
            return f"Error searching skills.sh: {e}"

        if not skills:
            return f'No skills found for "{query}".'

        lines: list[str] = [f'Found {len(skills)} skill(s) for "{query}":\n']
        for i, s in enumerate(skills, 1):
            name = s.get("name") or s.get("skillId") or "unknown"
            installs: int = s.get("installs", 0)
            skill_id_val: str = s.get("skillId") or ""
            source: str = s.get("source") or ""
            install_id = f"{source}/{skill_id_val}" if skill_id_val and source else source or skill_id_val
            lines.append(f"{i}. {name} ({installs:,} installs)")
            lines.append(f'   \u2192 install_skill("{install_id}")')
            lines.append("")
        return "\n".join(lines)

    return search_skills


def _make_install_tool(
    client: SkillsClient,
    lock: SkillsLock,
    runtime: SkillRuntime,
) -> FunctionTool:
    @tool
    async def install_skill(skill_id: str) -> str:
        """Download and install a skill from skills.sh.

        Args:
            skill_id: The skill identifier from search results, e.g.:
                      ``"vercel-labs/agent-skills/react-best-practices"`` (monorepo),
                      ``"mvanhorn/last30days-skill"`` (standalone repo).
        """
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

    return install_skill


def _make_remove_tool(runtime: SkillRuntime, lock: SkillsLock) -> FunctionTool:
    @tool
    def remove_skill(name: str) -> str:
        """Remove an installed skill by name.

        Args:
            name: Skill name as returned by list_skills().
        """
        try:
            runtime.remove(name)
        except (ValueError, FileNotFoundError) as e:
            return str(e)
        lock.remove(name)
        runtime.invalidate()
        return f"Removed: {name}"

    return remove_skill
