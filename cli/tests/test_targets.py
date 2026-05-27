"""Tests for install targets (base, claude, copilot)."""

from pathlib import Path

from ag2_cli.install.registry import ContentItem
from ag2_cli.install.targets.base import (
    AG2_MARKER,
    DirectoryTarget,
    SingleFileTarget,
    format_frontmatter,
)
from ag2_cli.install.targets.claude import ClaudeTarget
from ag2_cli.install.targets.copilot import CopilotTarget


def _make_item(
    name: str = "test-skill",
    description: str = "A test skill",
    category: str = "skill",
    frontmatter: dict | None = None,
    body: str = "Skill body content.",
) -> ContentItem:
    """Helper to create a ContentItem for testing."""
    return ContentItem(
        name=name,
        description=description,
        category=category,
        frontmatter=frontmatter or {},
        body=body,
    )


class TestFormatFrontmatter:
    """Test format_frontmatter utility."""

    def test_empty_dict_returns_empty_string(self):
        assert format_frontmatter({}) == ""

    def test_booleans_formatted_lowercase(self):
        result = format_frontmatter({"alwaysApply": True, "disabled": False})
        assert "alwaysApply: true" in result
        assert "disabled: false" in result
        assert result.startswith("---")
        assert result.endswith("---")

    def test_strings_with_special_chars_are_quoted(self):
        result = format_frontmatter({"globs": "**/*.py", "plain": "hello"})
        assert 'globs: "**/*.py"' in result
        assert "plain: hello" in result


class TestDirectoryTarget:
    """Test DirectoryTarget install and uninstall."""

    def test_install_creates_files_in_correct_directory(self, tmp_path: Path):
        target = DirectoryTarget(
            name="test",
            display_name="Test",
            rules_dir=".test/rules",
        )
        items = [
            _make_item(name="skill-a", body="Content A"),
            _make_item(name="skill-b", body="Content B"),
        ]

        paths = target.install(tmp_path, items)

        assert len(paths) == 2
        rules_dir = tmp_path / ".test" / "rules"
        assert rules_dir.is_dir()

        file_a = rules_dir / "ag2-skill-a.md"
        file_b = rules_dir / "ag2-skill-b.md"
        assert file_a.exists()
        assert file_b.exists()
        assert "Content A" in file_a.read_text()
        assert "Content B" in file_b.read_text()

    def test_uninstall_removes_files_with_correct_prefix(self, tmp_path: Path):
        rules_dir = tmp_path / ".test" / "rules"
        rules_dir.mkdir(parents=True)

        # Create AG2 files and a non-AG2 file
        (rules_dir / "ag2-skill-a.md").write_text("content a")
        (rules_dir / "ag2-skill-b.md").write_text("content b")
        (rules_dir / "user-rule.md").write_text("user content")

        target = DirectoryTarget(
            name="test",
            display_name="Test",
            rules_dir=".test/rules",
        )

        removed = target.uninstall(tmp_path)

        assert len(removed) == 2
        assert not (rules_dir / "ag2-skill-a.md").exists()
        assert not (rules_dir / "ag2-skill-b.md").exists()
        # User file untouched
        assert (rules_dir / "user-rule.md").exists()


class TestSingleFileTarget:
    """Test SingleFileTarget install and uninstall."""

    def test_install_creates_file_with_marker(self, tmp_path: Path):
        target = SingleFileTarget(
            name="test",
            display_name="Test",
            file_path="RULES.md",
        )
        items = [_make_item(name="my-skill", body="Skill body.")]

        paths = target.install(tmp_path, items)

        assert len(paths) == 1
        content = (tmp_path / "RULES.md").read_text()
        assert AG2_MARKER in content
        assert content.count(AG2_MARKER) == 2
        assert "Skill body." in content

    def test_install_replaces_existing_ag2_section(self, tmp_path: Path):
        target = SingleFileTarget(
            name="test",
            display_name="Test",
            file_path="RULES.md",
        )

        # Write initial content with an existing AG2 section
        rules_file = tmp_path / "RULES.md"
        rules_file.write_text(f"User rules here.\n\n{AG2_MARKER}\n# Old AG2 content\n{AG2_MARKER}\n")

        items = [_make_item(name="new-skill", body="New content.")]
        target.install(tmp_path, items)

        content = rules_file.read_text()
        assert "Old AG2 content" not in content
        assert "New content." in content
        assert content.count(AG2_MARKER) == 2
        assert content.startswith("User rules here.")

    def test_install_handles_single_marker_corrupted(self, tmp_path: Path):
        target = SingleFileTarget(
            name="test",
            display_name="Test",
            file_path="RULES.md",
        )

        # Simulate corrupted file with only one marker
        rules_file = tmp_path / "RULES.md"
        rules_file.write_text(f"User rules.\n\n{AG2_MARKER}\n# Broken section without closing marker")

        items = [_make_item(name="fixed-skill", body="Fixed content.")]
        target.install(tmp_path, items)

        content = rules_file.read_text()
        assert "Fixed content." in content
        assert content.count(AG2_MARKER) == 2
        assert content.startswith("User rules.")
        # Old broken content replaced
        assert "Broken section without closing marker" not in content

    def test_uninstall_removes_ag2_section(self, tmp_path: Path):
        target = SingleFileTarget(
            name="test",
            display_name="Test",
            file_path="RULES.md",
        )

        rules_file = tmp_path / "RULES.md"
        rules_file.write_text(f"User rules here.\n\n{AG2_MARKER}\n# AG2 stuff\n{AG2_MARKER}\n\nMore user content.")

        removed = target.uninstall(tmp_path)

        assert len(removed) == 1
        content = rules_file.read_text()
        assert AG2_MARKER not in content
        assert "User rules here." in content

    def test_uninstall_deletes_file_if_only_ag2_content(self, tmp_path: Path):
        target = SingleFileTarget(
            name="test",
            display_name="Test",
            file_path="RULES.md",
        )

        rules_file = tmp_path / "RULES.md"
        rules_file.write_text(f"{AG2_MARKER}\n# AG2 only content\n{AG2_MARKER}\n")

        removed = target.uninstall(tmp_path)

        assert len(removed) == 1
        assert not rules_file.exists()


class TestClaudeTarget:
    """Test ClaudeTarget install and uninstall."""

    def test_install_creates_skill_directory_structure(self, tmp_path: Path):
        target = ClaudeTarget()
        items = [_make_item(name="my-skill", description="My skill", category="skill", body="Do things.")]

        paths = target.install(tmp_path, items)

        assert len(paths) == 1
        skill_file = tmp_path / ".claude" / "skills" / "ag2-my-skill" / "SKILL.md"
        assert skill_file.exists()
        content = skill_file.read_text()
        assert "Do things." in content
        assert "ag2-my-skill" in content
        assert "user_invocable: true" in content

    def test_install_creates_command_files(self, tmp_path: Path):
        target = ClaudeTarget()
        items = [
            _make_item(name="run-tests", description="Run the test suite", category="command", body="Execute tests.")
        ]

        paths = target.install(tmp_path, items)

        assert len(paths) == 1
        cmd_file = tmp_path / ".claude" / "commands" / "ag2-run-tests.md"
        assert cmd_file.exists()
        content = cmd_file.read_text()
        assert "# run-tests" in content
        assert "Run the test suite" in content
        assert "Execute tests." in content

    def test_uninstall_removes_ag2_prefixed_skill_dirs(self, tmp_path: Path):
        target = ClaudeTarget()

        # Create skill dirs (with nested files) and a non-AG2 dir
        skills_dir = tmp_path / ".claude" / "skills"
        skills_dir.mkdir(parents=True)

        ag2_skill = skills_dir / "ag2-my-skill"
        ag2_skill.mkdir()
        (ag2_skill / "SKILL.md").write_text("skill content")
        (ag2_skill / "extra.txt").write_text("extra file")

        ag2_other = skills_dir / "ag2-other"
        ag2_other.mkdir()
        (ag2_other / "SKILL.md").write_text("other content")

        user_skill = skills_dir / "user-skill"
        user_skill.mkdir()
        (user_skill / "SKILL.md").write_text("user skill")

        # Create command files
        cmd_dir = tmp_path / ".claude" / "commands"
        cmd_dir.mkdir(parents=True)
        (cmd_dir / "ag2-run-tests.md").write_text("cmd content")
        (cmd_dir / "user-cmd.md").write_text("user cmd")

        removed = target.uninstall(tmp_path)

        # AG2 skill dirs removed (including nested files)
        assert not ag2_skill.exists()
        assert not ag2_other.exists()
        # User skill untouched
        assert user_skill.exists()
        # AG2 command removed, user command untouched
        assert not (cmd_dir / "ag2-run-tests.md").exists()
        assert (cmd_dir / "user-cmd.md").exists()

        # Removed list includes files from inside dirs + the dirs + the command file
        removed_strs = [str(p) for p in removed]
        assert any("ag2-my-skill" in s and "SKILL.md" in s for s in removed_strs)
        assert any("ag2-run-tests.md" in s for s in removed_strs)


class TestCopilotTarget:
    """Test CopilotTarget install and uninstall."""

    def test_install_creates_instruction_files(self, tmp_path: Path):
        target = CopilotTarget()
        items = [
            _make_item(
                name="my-rule",
                description="A rule",
                category="rule",
                frontmatter={"globs": "**/*.py"},
                body="Rule content.",
            )
        ]

        paths = target.install(tmp_path, items)

        assert len(paths) == 1
        inst_file = tmp_path / ".github" / "instructions" / "ag2-my-rule.instructions.md"
        assert inst_file.exists()
        content = inst_file.read_text()
        assert "Rule content." in content
        assert "applyTo" in content

    def test_uninstall_removes_instruction_files(self, tmp_path: Path):
        target = CopilotTarget()

        inst_dir = tmp_path / ".github" / "instructions"
        inst_dir.mkdir(parents=True)
        (inst_dir / "ag2-my-rule.instructions.md").write_text("ag2 content")
        (inst_dir / "ag2-other.instructions.md").write_text("ag2 other")
        (inst_dir / "user-rule.instructions.md").write_text("user content")

        removed = target.uninstall(tmp_path)

        assert len(removed) == 2
        assert not (inst_dir / "ag2-my-rule.instructions.md").exists()
        assert not (inst_dir / "ag2-other.instructions.md").exists()
        assert (inst_dir / "user-rule.instructions.md").exists()


class TestDetectTargets:
    """Test target detection from project directory."""

    def test_detects_correct_targets(self, tmp_path: Path):
        from ag2_cli.install.targets import detect_targets

        # Create markers for Claude and Copilot
        (tmp_path / "CLAUDE.md").write_text("claude")
        (tmp_path / ".github").mkdir()

        detected = detect_targets(tmp_path)
        detected_names = [t.name for t in detected]

        assert "claude" in detected_names
        assert "copilot" in detected_names
        # Cursor should not be detected
        assert "cursor" not in detected_names


class TestGetTargetAndGetAllTargets:
    """Test target lookup functions."""

    def test_get_target_returns_known_target(self):
        from ag2_cli.install.targets import get_target

        target = get_target("claude")
        assert target is not None
        assert target.name == "claude"
        assert target.display_name == "Claude Code"

    def test_get_target_returns_none_for_unknown(self):
        from ag2_cli.install.targets import get_target

        assert get_target("nonexistent-target") is None

    def test_get_all_targets_returns_all(self):
        from ag2_cli.install.targets import get_all_targets

        targets = get_all_targets()
        assert len(targets) > 0
        names = [t.name for t in targets]
        assert "claude" in names
        assert "copilot" in names
        assert "cursor" in names
