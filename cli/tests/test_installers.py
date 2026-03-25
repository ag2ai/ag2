"""Tests for artifact installers."""

import json
from pathlib import Path


class TestSkillsInstaller:
    """Test the skills installer with bundled content."""

    def test_install_bundled_ag2_skills(self, tmp_path: Path):
        from ag2_cli.install.client import ArtifactClient
        from ag2_cli.install.installers.skills import SkillsInstaller
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver
        from ag2_cli.install.targets import get_target

        client = ArtifactClient(cache_dir=tmp_path / "cache")
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        installer = SkillsInstaller(client, lockfile, resolver)

        # Use the Claude target for testing
        claude = get_target("claude")
        assert claude is not None

        results = installer.install(["ag2"], [claude], tmp_path)
        assert len(results) == 1
        result = results[0]
        assert result.artifact.name == "ag2"
        assert len(result.files_created) > 0

        # Verify files were created
        skills_dir = tmp_path / ".claude" / "skills"
        assert skills_dir.exists()
        skill_dirs = list(skills_dir.iterdir())
        assert len(skill_dirs) > 0

        # Verify lockfile was updated
        assert lockfile.is_installed("skills/ag2ai/ag2")

    def test_install_with_name_filter(self, tmp_path: Path):
        from ag2_cli.install.client import ArtifactClient
        from ag2_cli.install.installers.skills import SkillsInstaller
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver
        from ag2_cli.install.targets import get_target

        client = ArtifactClient(cache_dir=tmp_path / "cache")
        lockfile = Lockfile(tmp_path)
        resolver = DependencyResolver(client, lockfile)

        installer = SkillsInstaller(client, lockfile, resolver)
        claude = get_target("claude")

        results = installer.install(["ag2"], [claude], tmp_path, name_filter="imports")
        assert len(results) == 1
        # Should only install the "imports" item
        result = results[0]
        assert len(result.files_created) > 0
        # Claude target creates .claude/skills/ag2-imports/SKILL.md
        file_parents = [f.parent.name for f in result.files_created]
        assert any("imports" in p for p in file_parents)


class TestLoadSkillsFromArtifact:
    """Test loading skills from both flat and directory formats."""

    def test_loads_flat_format(self, tmp_path: Path):
        from ag2_cli.install.artifact import Artifact
        from ag2_cli.install.installers.skills import load_skills_from_artifact

        # Create flat format skills
        rules_dir = tmp_path / "rules"
        rules_dir.mkdir()
        (rules_dir / "my-rule.md").write_text("---\nname: my-rule\ndescription: A test rule\n---\n\nRule content here.")

        artifact = Artifact(name="test", type="skills", source_dir=tmp_path)
        items = load_skills_from_artifact(artifact)

        assert len(items) == 1
        assert items[0].name == "my-rule"
        assert items[0].category == "rule"
        assert "Rule content here." in items[0].body

    def test_loads_directory_format(self, tmp_path: Path):
        from ag2_cli.install.artifact import Artifact
        from ag2_cli.install.installers.skills import load_skills_from_artifact

        # Create Agent Skills standard format
        skill_dir = tmp_path / "skills" / "build-chat"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: build-chat\ndescription: Build a group chat\n---\n\nSkill content."
        )

        artifact = Artifact(name="test", type="skills", source_dir=tmp_path)
        items = load_skills_from_artifact(artifact)

        assert len(items) == 1
        assert items[0].name == "build-chat"
        assert items[0].category == "skill"

    def test_loads_mixed_categories(self, tmp_path: Path):
        from ag2_cli.install.artifact import Artifact
        from ag2_cli.install.installers.skills import load_skills_from_artifact

        for cat, subdir in [("rule", "rules"), ("skill", "skills"), ("agent", "agents"), ("command", "commands")]:
            d = tmp_path / subdir
            d.mkdir()
            (d / f"test-{cat}.md").write_text(f"---\nname: test-{cat}\ndescription: A {cat}\n---\n\nContent.")

        artifact = Artifact(name="test", type="skills", source_dir=tmp_path)
        items = load_skills_from_artifact(artifact)

        assert len(items) == 4
        categories = {i.category for i in items}
        assert categories == {"rule", "skill", "agent", "command"}


class TestTemplateInstaller:
    """Test template scaffolding."""

    def _make_template(self, tmp_path: Path) -> Path:
        """Create a minimal template artifact in tmp_path/cache."""
        template_dir = tmp_path / "cache" / "templates" / "ag2ai" / "test-app" / "latest"
        template_dir.mkdir(parents=True)

        manifest = {
            "name": "test-app",
            "type": "template",
            "display_name": "Test App",
            "version": "1.0.0",
            "template": {
                "scaffold": "scaffold/",
                "variables": {
                    "project_name": {"prompt": "Name", "default": "my-app"},
                },
                "ignore": ["*.pyc"],
                "post_install": [],
            },
        }
        (template_dir / "artifact.json").write_text(json.dumps(manifest))

        # Create scaffold
        scaffold = template_dir / "scaffold"
        scaffold.mkdir()
        (scaffold / "README.md.tmpl").write_text("# {{ project_name }}\n\nWelcome.")
        (scaffold / "main.py").write_text("print('hello')")
        (scaffold / "src").mkdir()
        (scaffold / "src" / "app.py").write_text("# {{ project_name }} app")

        # Add .fetched marker
        (template_dir / ".fetched").touch()

        return template_dir

    def test_scaffolds_project(self, tmp_path: Path):
        from ag2_cli.install.client import ArtifactClient
        from ag2_cli.install.installers.skills import SkillsInstaller
        from ag2_cli.install.installers.templates import TemplateInstaller
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        self._make_template(tmp_path)
        project = tmp_path / "project"
        project.mkdir()

        client = ArtifactClient(cache_dir=tmp_path / "cache")
        lockfile = Lockfile(project)
        resolver = DependencyResolver(client, lockfile)
        skills = SkillsInstaller(client, lockfile, resolver)
        installer = TemplateInstaller(client, lockfile, resolver, skills)

        installer.install("test-app", project, [], variables={"project_name": "my-cool-app"})

        # Check scaffold was copied
        assert (project / "main.py").exists()
        assert (project / "src" / "app.py").exists()

        # Check variable substitution in .tmpl file
        readme = (project / "README.md").read_text()
        assert "# my-cool-app" in readme
        assert "{{ project_name }}" not in readme

        # .tmpl extension should be stripped
        assert not (project / "README.md.tmpl").exists()

    def test_preview_mode(self, tmp_path: Path):
        from ag2_cli.install.client import ArtifactClient
        from ag2_cli.install.installers.skills import SkillsInstaller
        from ag2_cli.install.installers.templates import TemplateInstaller
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        self._make_template(tmp_path)
        project = tmp_path / "project"
        project.mkdir()

        client = ArtifactClient(cache_dir=tmp_path / "cache")
        lockfile = Lockfile(project)
        resolver = DependencyResolver(client, lockfile)
        skills = SkillsInstaller(client, lockfile, resolver)
        installer = TemplateInstaller(client, lockfile, resolver, skills)

        result = installer.install("test-app", project, [], preview=True)

        # Files listed but not created
        assert len(result.files_created) > 0
        assert "Preview mode" in result.warnings[0]
        assert not (project / "main.py").exists()


class TestAgentInstaller:
    """Test agent installation to Claude Code."""

    def test_installs_agent_file(self, tmp_path: Path):
        from ag2_cli.install.client import ArtifactClient
        from ag2_cli.install.installers.agents import AgentInstaller
        from ag2_cli.install.installers.skills import SkillsInstaller
        from ag2_cli.install.lockfile import Lockfile
        from ag2_cli.install.resolver import DependencyResolver

        # Create agent artifact
        agent_dir = tmp_path / "cache" / "agents" / "ag2ai" / "test-agent" / "latest"
        agent_dir.mkdir(parents=True)
        manifest = {
            "name": "test-agent",
            "type": "agent",
            "version": "1.0.0",
            "agent": {
                "source": "agent.md",
                "model": "sonnet",
                "tools": ["Read", "Grep"],
                "mcp_servers": {},
            },
        }
        (agent_dir / "artifact.json").write_text(json.dumps(manifest))
        (agent_dir / "agent.md").write_text(
            "---\nname: ag2-test-agent\ndescription: Test agent\n---\n\nYou are a test agent."
        )
        (agent_dir / ".fetched").touch()

        project = tmp_path / "project"
        project.mkdir()

        client = ArtifactClient(cache_dir=tmp_path / "cache")
        lockfile = Lockfile(project)
        resolver = DependencyResolver(client, lockfile)
        skills = SkillsInstaller(client, lockfile, resolver)
        installer = AgentInstaller(client, lockfile, resolver, skills)

        installer.install("test-agent", project, [])

        # Check agent file installed
        agent_file = project / ".claude" / "agents" / "ag2-test-agent.md"
        assert agent_file.exists()
        content = agent_file.read_text()
        assert "You are a test agent." in content

        # Check lockfile
        assert lockfile.is_installed("agents/ag2ai/test-agent")
