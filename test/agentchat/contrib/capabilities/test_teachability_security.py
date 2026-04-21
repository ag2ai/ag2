# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Security tests for teachability.py MemoStore -- R4R1-B1.

Verifies that the legacy pickle store is rejected and the JSON store
round-trips correctly.
"""

import json
import os
import pickle
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_memo_store(path: str, reset: bool = False):
    """Construct a MemoStore, mocking chromadb to avoid the dependency."""
    from unittest.mock import MagicMock, patch

    with patch("autogen.agentchat.contrib.capabilities.teachability.chromadb") as mock_chromadb:
        mock_vec_db = MagicMock()
        mock_db_client = MagicMock()
        mock_db_client.create_collection.return_value = mock_vec_db
        mock_chromadb.Client.return_value = mock_db_client

        from autogen.agentchat.contrib.capabilities.teachability import MemoStore

        return MemoStore(verbosity=0, reset=reset, path_to_db_dir=path)


# ---------------------------------------------------------------------------
# R4R1-B1: pickle rejection and JSON migration gate
# ---------------------------------------------------------------------------


class TestMemoStorePickleRejection:
    def test_legacy_pkl_only_raises_runtime_error(self):
        """If only uid_text_dict.pkl exists, MemoStore MUST raise RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "uid_text_dict.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({"1": ("input", "output")}, f)

            with pytest.raises(RuntimeError, match="teachability_migrate_pickle_to_json"):
                _make_memo_store(tmpdir)

    def test_adversarial_pickle_rce_payload_in_pkl_file(self):
        """Attacker-writable .pkl with RCE payload MUST be rejected at load time."""

        class _Exploit:
            def __reduce__(self):
                return (os.system, ("echo PWNED",))

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "uid_text_dict.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(_Exploit(), f)

            with pytest.raises(RuntimeError, match="teachability_migrate_pickle_to_json"):
                _make_memo_store(tmpdir)

    def test_json_store_loads_correctly(self):
        """If uid_text_dict.json exists, MemoStore MUST load it without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            data = {"1": ["hello", "world"], "2": ["foo", "bar"]}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            store = _make_memo_store(tmpdir)
            assert store.uid_text_dict == data
            assert store.last_memo_id == 2

    def test_json_store_created_fresh_on_new_dir(self):
        """MemoStore MUST create an empty JSON store for a new directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_memo_store(tmpdir)
            assert store.uid_text_dict == {}
            assert store.last_memo_id == 0

    def test_save_memos_writes_json_not_pickle(self):
        """_save_memos() MUST write valid JSON, not pickle bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = _make_memo_store(tmpdir)
            store.uid_text_dict = {"1": ["a", "b"]}
            store._save_memos()

            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            assert os.path.exists(json_path)

            with open(json_path, encoding="utf-8") as f:
                content = f.read()
            # Must be valid JSON, not pickle bytes
            parsed = json.loads(content)
            assert parsed == {"1": ["a", "b"]}

            # Verify it is definitely not a pickle file
            assert not content.startswith("\x80")

    def test_pkl_coexists_with_json_no_error(self):
        """If both .pkl and .json exist, the .json MUST be used without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a json store (already migrated)
            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"1": ["q", "a"]}, f)
            # Also write the legacy pkl (would exist after migration bak rename)
            pkl_path = os.path.join(tmpdir, "uid_text_dict.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump({}, f)

            # Should succeed because json takes precedence
            store = _make_memo_store(tmpdir)
            assert "1" in store.uid_text_dict

    def test_reset_clears_existing_store(self):
        """reset=True MUST clear the store even if a JSON file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({"1": ["a", "b"]}, f)

            store = _make_memo_store(tmpdir, reset=True)
            assert store.uid_text_dict == {}

    def test_invalid_json_schema_raises(self):
        """Non-dict JSON content MUST raise ValueError, not load silently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([1, 2, 3], f)  # list instead of dict

            with pytest.raises(ValueError, match="unexpected format"):
                _make_memo_store(tmpdir)


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------


class TestTeachabilityMigrationHelper:
    def test_migration_converts_pkl_to_json(self):
        """migrate() MUST produce a valid JSON file and rename the pkl."""
        from autogen.agentchat.contrib.capabilities.teachability_migrate_pickle_to_json import migrate

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, "uid_text_dict.pkl")
            data = {"1": ("question", "answer")}
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)

            migrate(tmpdir)

            json_path = os.path.join(tmpdir, "uid_text_dict.json")
            bak_path = pkl_path + ".bak"
            assert os.path.exists(json_path)
            assert os.path.exists(bak_path)
            assert not os.path.exists(pkl_path)

            with open(json_path, encoding="utf-8") as f:
                result = json.load(f)
            assert result == {"1": ["question", "answer"]}

    def test_migration_no_op_when_pkl_absent(self):
        """migrate() MUST print message and return cleanly when no .pkl present."""
        from autogen.agentchat.contrib.capabilities.teachability_migrate_pickle_to_json import migrate

        with tempfile.TemporaryDirectory() as tmpdir:
            # No pkl file -- should not raise
            migrate(tmpdir)
