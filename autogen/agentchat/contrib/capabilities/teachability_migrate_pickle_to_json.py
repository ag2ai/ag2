# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Migration helper: convert a legacy pickle-based teachability store to JSON.

Usage:
    python -m autogen.agentchat.contrib.capabilities.teachability_migrate_pickle_to_json \\
        --path /path/to/db_dir

The script reads uid_text_dict.pkl, validates the contents, writes
uid_text_dict.json, then renames the old file to uid_text_dict.pkl.bak.

Run ONCE per store directory before upgrading to the new teachability version.
"""

import argparse
import json
import os
import pickle
import shutil
import sys


def migrate(path_to_db_dir: str) -> None:
    """Migrate a pickle memo store to JSON in-place.

    Args:
        path_to_db_dir: Directory containing uid_text_dict.pkl.
    """
    pkl_path = os.path.join(path_to_db_dir, "uid_text_dict.pkl")
    json_path = os.path.join(path_to_db_dir, "uid_text_dict.json")
    bak_path = pkl_path + ".bak"

    if not os.path.exists(pkl_path):
        print(f"No legacy store found at {pkl_path!r}. Nothing to migrate.")
        return

    if os.path.exists(json_path):
        print(f"JSON store already exists at {json_path!r}. Remove it first if you want to re-migrate.")
        sys.exit(1)

    print(f"Reading pickle store from {pkl_path!r} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301 -- user-controlled migration, not network data

    if not isinstance(data, dict):
        print(f"ERROR: Expected dict, got {type(data).__name__}. Aborting.")
        sys.exit(1)

    print(f"Writing JSON store to {json_path!r} ...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Renaming {pkl_path!r} to {bak_path!r} ...")
    shutil.move(pkl_path, bak_path)

    print(f"Migration complete. {len(data)} entries written.")
    print(f"Old pickle file kept as {bak_path!r} -- delete it when confident.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate teachability store from pickle to JSON.")
    parser.add_argument("--path", required=True, help="Path to the teachability DB directory.")
    args = parser.parse_args()
    migrate(args.path)


if __name__ == "__main__":
    main()
