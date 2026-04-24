"""Tests for the lockfile system."""

from pathlib import Path


class TestLockfile:
    """Test lockfile operations."""

    def test_empty_lockfile(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        assert lock.list_installed() == []
        assert lock.is_installed("skills/ag2") is False

    def test_record_and_check_install(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        lock.record_install(
            ref="skills/ag2",
            version="0.3.0",
            targets=["claude", "cursor"],
            files=[tmp_path / "file1.md", tmp_path / "file2.md"],
        )

        assert lock.is_installed("skills/ag2")
        assert lock.is_installed("skills/ag2", version="0.3.0")
        assert lock.is_installed("skills/ag2", version="0.4.0") is False
        assert lock.is_installed("skills/fastapi") is False

    def test_lockfile_persists_to_disk(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock1 = Lockfile(tmp_path)
        lock1.record_install(ref="tools/web-search", version="1.0.0", targets=["claude"], files=[])

        # Load from disk
        lock2 = Lockfile(tmp_path)
        assert lock2.is_installed("tools/web-search")
        info = lock2.get_installed("tools/web-search")
        assert info is not None
        assert info.version == "1.0.0"

    def test_record_uninstall(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        lock.record_install(ref="skills/ag2", version="0.3.0", targets=["claude"], files=[])

        info = lock.record_uninstall("skills/ag2")
        assert info is not None
        assert info.ref == "skills/ag2"
        assert lock.is_installed("skills/ag2") is False

    def test_uninstall_nonexistent_returns_none(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        assert lock.record_uninstall("skills/nonexistent") is None

    def test_list_installed(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        lock.record_install(ref="skills/ag2", version="0.3.0", targets=["claude"], files=[])
        lock.record_install(ref="tools/web-search", version="1.0.0", targets=["cursor"], files=[])

        installed = lock.list_installed()
        assert len(installed) == 2
        refs = {i.ref for i in installed}
        assert refs == {"skills/ag2", "tools/web-search"}

    def test_overwrite_existing_install(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        lock.record_install(ref="skills/ag2", version="0.3.0", targets=["claude"], files=[])
        lock.record_install(ref="skills/ag2", version="0.4.0", targets=["claude", "cursor"], files=[])

        assert lock.is_installed("skills/ag2", version="0.4.0")
        assert lock.is_installed("skills/ag2", version="0.3.0") is False
        assert len(lock.list_installed()) == 1

    def test_files_stored_as_relative(self, tmp_path: Path):
        from ag2_cli.install.lockfile import Lockfile

        lock = Lockfile(tmp_path)
        lock.record_install(
            ref="skills/ag2",
            version="0.3.0",
            targets=["claude"],
            files=[tmp_path / ".claude" / "skills" / "ag2-test" / "SKILL.md"],
        )

        info = lock.get_installed("skills/ag2")
        assert info is not None
        assert info.files == [".claude/skills/ag2-test/SKILL.md"]
