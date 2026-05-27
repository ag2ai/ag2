"""Tests for the content pack registry module."""

import json
from pathlib import Path


class TestParseFrontmatter:
    """Test parse_frontmatter with various inputs."""

    def test_valid_frontmatter(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nname: my-skill\ndescription: A cool skill\n---\n\nBody content here."
        fm, body = parse_frontmatter(text)

        assert fm["name"] == "my-skill"
        assert fm["description"] == "A cool skill"
        assert body == "Body content here."

    def test_no_frontmatter_returns_empty_dict_and_full_text(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "Just some plain text\nwith multiple lines."
        fm, body = parse_frontmatter(text)

        assert fm == {}
        assert body == text

    def test_missing_closing_delimiter_returns_empty_dict_and_full_text(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nname: broken\ndescription: No closing\n\nBody here."
        fm, body = parse_frontmatter(text)

        assert fm == {}
        assert body == text

    def test_boolean_values_converted_to_python_bools(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nalwaysApply: true\ndisabled: false\nMixed: True\nAlso: FALSE\n---\n\nContent."
        fm, body = parse_frontmatter(text)

        assert fm["alwaysApply"] is True
        assert fm["disabled"] is False
        assert fm["Mixed"] is True
        assert fm["Also"] is False

    def test_quoted_strings_have_quotes_stripped(self):
        from ag2_cli.install.registry import parse_frontmatter

        text = "---\nname: \"my-skill\"\ndescription: 'A quoted description'\n---\n\nBody."
        fm, body = parse_frontmatter(text)

        assert fm["name"] == "my-skill"
        assert fm["description"] == "A quoted description"


class TestLoadPack:
    """Test load_pack with real filesystem fixtures."""

    def test_returns_none_for_nonexistent_pack(self, tmp_path: Path, monkeypatch):
        from ag2_cli.install import registry

        monkeypatch.setattr(registry, "CONTENT_DIR", tmp_path)

        result = registry.load_pack("nonexistent")
        assert result is None

    def test_returns_none_when_no_manifest(self, tmp_path: Path, monkeypatch):
        from ag2_cli.install import registry

        monkeypatch.setattr(registry, "CONTENT_DIR", tmp_path)

        # Create a directory but no manifest.json inside it
        pack_dir = tmp_path / "my-pack"
        pack_dir.mkdir()

        result = registry.load_pack("my-pack")
        assert result is None


class TestListPacks:
    """Test list_packs discovery."""

    def test_returns_pack_names(self, tmp_path: Path, monkeypatch):
        from ag2_cli.install import registry

        monkeypatch.setattr(registry, "CONTENT_DIR", tmp_path)

        # Create two valid packs and one invalid (no manifest)
        for name in ["alpha", "beta"]:
            d = tmp_path / name
            d.mkdir()
            manifest = {
                "name": name,
                "display_name": name.title(),
                "description": f"{name} pack",
                "version": "1.0.0",
            }
            (d / "manifest.json").write_text(json.dumps(manifest))

        # Directory without manifest should be excluded
        (tmp_path / "no-manifest").mkdir()

        packs = registry.list_packs()
        assert sorted(packs) == ["alpha", "beta"]
