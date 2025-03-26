# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0


import textwrap
from pathlib import Path

from autogen._website.utils import copy_files, remove_marker_blocks


def test_copy_files(tmp_path: Path) -> None:
    # Create source directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    # Create some test files in source
    test_files = [src_dir / "file1.txt", src_dir / "subdir" / "file2.txt"]

    # Create the subdir and files
    (src_dir / "subdir").mkdir(parents=True)
    for file in test_files:
        file.write_text("test content")

    # Create destination directory
    dst_dir = tmp_path / "dst"
    dst_dir.mkdir()

    # Call the function
    copy_files(src_dir, dst_dir, test_files)

    # Verify files were copied correctly
    for src_file in test_files:
        dst_file = dst_dir / src_file.relative_to(src_dir)
        assert dst_file.exists()
        assert dst_file.read_text() == "test content"


def test_remove_marker_blocks() -> None:
    content = textwrap.dedent("""
        **Contribution guide:**
        Built something interesting with AG2? Submit a PR to add it to the list! See the [Contribution Guide below](#contributing) for more details.

        {/* DELETE-ME-WHILE-BUILDING-MKDOCS-START */}

        <ClientSideComponent Component={GalleryPage} componentProps={{galleryItems: galleryItems}} />

        {/* DELETE-ME-WHILE-BUILDING-MKDOCS-END */}

        {/* DELETE-ME-WHILE-BUILDING-MINTLIFY-START */}

        {{ render_gallery(gallery_items) }}

        {/* DELETE-ME-WHILE-BUILDING-MINTLIFY-END */}

        ## Contributing
        Thank you for your interest in contributing! To add your demo to the gallery.""")

    expected = textwrap.dedent("""
        **Contribution guide:**
        Built something interesting with AG2? Submit a PR to add it to the list! See the [Contribution Guide below](#contributing) for more details.

        <ClientSideComponent Component={GalleryPage} componentProps={{galleryItems: galleryItems}} />

        ## Contributing
        Thank you for your interest in contributing! To add your demo to the gallery.""")

    # Test removing Mintlify blocks (for MkDocs build)
    actual = remove_marker_blocks(content, "DELETE-ME-WHILE-BUILDING-MINTLIFY")
    assert actual == expected

    expected = textwrap.dedent("""
        **Contribution guide:**
        Built something interesting with AG2? Submit a PR to add it to the list! See the [Contribution Guide below](#contributing) for more details.

        {{ render_gallery(gallery_items) }}

        ## Contributing
        Thank you for your interest in contributing! To add your demo to the gallery.""")

    # Test removing Mintlify blocks (for MkDocs build)
    actual = remove_marker_blocks(content, "DELETE-ME-WHILE-BUILDING-MKDOCS")
    assert actual == expected
