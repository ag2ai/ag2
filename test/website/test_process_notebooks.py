# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Generator, Optional, Union

import pytest

# Add the ../../website directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "website"))
from process_notebooks import (
    add_authors_and_social_img_to_blog_posts,
    add_front_matter_to_metadata_mdx,
    cleanup_tmp_dirs_if_no_metadata,
    convert_callout_blocks,
    ensure_mint_json_exists,
    extract_example_group,
    generate_nav_group,
    get_sorted_files,
)


def test_ensure_mint_json() -> None:
    # Test with empty temp directory - should raise SystemExit
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with pytest.raises(SystemExit):
            ensure_mint_json_exists(tmp_path)

        # Now create mint.json - should not raise error
        (tmp_path / "mint.json").touch()
        ensure_mint_json_exists(tmp_path)  # Should not raise any exception


def test_cleanup_tmp_dirs_if_no_metadata() -> None:
    # Test without the tmp_dir / "snippets" / "data" / "NotebooksMetadata.mdx"
    # the tmp_dir / "notebooks" should be removed.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        (notebooks_dir / "example-1.mdx").touch()
        (notebooks_dir / "example-2.mdx").touch()

        cleanup_tmp_dirs_if_no_metadata(tmp_path)
        assert not notebooks_dir.exists()

    # Test with the tmp_dir / "snippets" / "data" / "NotebooksMetadata.mdx"
    # the tmp_dir / "notebooks" should not be removed.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        notebooks_dir = tmp_path / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        (notebooks_dir / "example-1.mdx").touch()
        (notebooks_dir / "example-2.mdx").touch()

        metadata_dir = tmp_path / "snippets" / "data"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "NotebooksMetadata.mdx").touch()

        cleanup_tmp_dirs_if_no_metadata(tmp_path)
        assert notebooks_dir.exists()


class TestAddFrontMatterToMetadataMdx:
    def test_without_metadata_mdx(self) -> None:
        front_matter_dict: dict[str, Union[str, Optional[Union[list[str]]]]] = {
            "title": "some title",
            "link": "/notebooks/some-title",
            "description": "some description",
            "image": "some image",
            "tags": ["tag1", "tag2"],
            "source_notebook": "/notebook/some-title.ipynb",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            metadata_dir = tmp_path / "snippets" / "data"
            metadata_dir.mkdir(parents=True, exist_ok=True)

            rendered_mdx = tmp_path / "source"
            rendered_mdx.mkdir(parents=True, exist_ok=True)
            (rendered_mdx / "some-title.mdx").touch()

            # when the metadata file is not present, the file should be created and the front matter should be added
            add_front_matter_to_metadata_mdx(front_matter_dict, tmp_path, rendered_mdx)

            assert (metadata_dir / "NotebooksMetadata.mdx").exists()

            with open(metadata_dir / "NotebooksMetadata.mdx", "r") as f:
                actual = f.read()

            assert (
                actual
                == """{/*
Auto-generated file - DO NOT EDIT
Please edit the add_front_matter_to_metadata_mdx function in process_notebooks.py
*/}

export const notebooksMetadata = [
    {
        "title": "some title",
        "link": "/notebooks/source",
        "description": "some description",
        "image": "some image",
        "tags": [
            "tag1",
            "tag2"
        ],
        "source": "/notebook/some-title.ipynb"
    }
];
"""
            )

    def test_with_metadata_mdx(self) -> None:
        front_matter_dict: dict[str, Optional[Union[str, Union[list[str]]]]] = {
            "title": "some title",
            "link": "/notebooks/some-title",
            "description": "some description",
            "image": "some image",
            "tags": ["tag1", "tag2"],
            "source_notebook": "/notebook/some-title.ipynb",
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            metadata_dir = tmp_path / "snippets" / "data"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_content = """{/*
Auto-generated file - DO NOT EDIT
Please edit the add_front_matter_to_metadata_mdx function in process_notebooks.py
*/}

export const notebooksMetadata = [
    {
        "title": "some other title",
        "link": "/notebooks/some-other-title",
        "description": "some other description",
        "image": "some other image",
        "tags": [
            "tag3",
            "tag4"
        ],
        "source": "/notebook/some-other-title.ipynb"
    }
];
"""
            with open(metadata_dir / "NotebooksMetadata.mdx", "w") as f:
                f.write(metadata_content)

            rendered_mdx = tmp_path / "source"
            rendered_mdx.mkdir(parents=True, exist_ok=True)
            (rendered_mdx / "some-title.mdx").touch()

            # when the metadata file is present, the front matter should be added to the existing metadata
            add_front_matter_to_metadata_mdx(front_matter_dict, tmp_path, rendered_mdx)

            assert (metadata_dir / "NotebooksMetadata.mdx").exists()

            with open(metadata_dir / "NotebooksMetadata.mdx", "r") as f:
                actual = f.read()

            assert (
                actual
                == """{/*
Auto-generated file - DO NOT EDIT
Please edit the add_front_matter_to_metadata_mdx function in process_notebooks.py
*/}

export const notebooksMetadata = [
    {
        "title": "some other title",
        "link": "/notebooks/some-other-title",
        "description": "some other description",
        "image": "some other image",
        "tags": [
            "tag3",
            "tag4"
        ],
        "source": "/notebook/some-other-title.ipynb"
    },
    {
        "title": "some title",
        "link": "/notebooks/source",
        "description": "some description",
        "image": "some image",
        "tags": [
            "tag1",
            "tag2"
        ],
        "source": "/notebook/some-title.ipynb"
    }
];
"""
            )


class TestAddBlogsToNavigation:
    @pytest.fixture
    def test_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create test directory structure
            files = [
                "2023-04-21-LLM-tuning-math/index.mdx",
                "2023-05-18-GPT-adaptive-humaneval/index.mdx",
                "2023-10-26-TeachableAgent/index.mdx",
                "2023-10-18-RetrieveChat/index.mdx",
                "2024-12-20-RetrieveChat/index.mdx",
                "2024-12-20-Tools-interoperability/index.mdx",
                "2024-12-20-RealtimeAgent/index.mdx",
                "2024-08-26/index.mdx",
                "2024-11-11/index.mdx",
                "2024-11-12/index.mdx",
                "2024-12-20-Another-RealtimeAgent/index.mdx",
            ]

            for file_path in files:
                full_path = tmp_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            yield tmp_path

    @pytest.fixture
    def expected(self) -> list[str]:
        return [
            "blog/2024-12-20-Tools-interoperability/index",
            "blog/2024-12-20-RetrieveChat/index",
            "blog/2024-12-20-RealtimeAgent/index",
            "blog/2024-12-20-Another-RealtimeAgent/index",
            "blog/2024-11-12/index",
            "blog/2024-11-11/index",
            "blog/2024-08-26/index",
            "blog/2023-10-26-TeachableAgent/index",
            "blog/2023-10-18-RetrieveChat/index",
            "blog/2023-05-18-GPT-adaptive-humaneval/index",
            "blog/2023-04-21-LLM-tuning-math/index",
        ]

    def test_get_sorted_files(self, test_dir: Path, expected: list[str]) -> None:
        actual = get_sorted_files(test_dir, "blog")
        assert actual == expected, actual

    def test_add_blogs_to_navigation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            website_dir = Path(tmp_dir)
            blog_dir = website_dir / "blog"

            # Create test blog structure
            test_files = [
                "2023-04-21-LLM-tuning-math/index.mdx",
                "2024-12-20-RealtimeAgent/index.mdx",
                "2024-12-20-RetrieveChat/index.mdx",
            ]

            for file_path in test_files:
                full_path = blog_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.touch()

            # Expected result after processing
            expected = {
                "group": "Blog",
                "pages": [
                    "blog/2024-12-20-RetrieveChat/index",
                    "blog/2024-12-20-RealtimeAgent/index",
                    "blog/2023-04-21-LLM-tuning-math/index",
                ],
            }

            # Run function and check result
            actual = generate_nav_group(blog_dir, "Blog", "blog")
            assert actual == expected, actual

            # Expected result after processing
            expected = {
                "group": "Talks",
                "pages": [
                    "talks/2024-12-20-RetrieveChat/index",
                    "talks/2024-12-20-RealtimeAgent/index",
                    "talks/2023-04-21-LLM-tuning-math/index",
                ],
            }

            # Run function and check result
            actual = generate_nav_group(blog_dir, "Talks", "talks")
            assert actual == expected, actual


class TestUpdateNavigation:
    def setup(self, temp_dir: Path) -> None:
        """Set up test files in the temporary directory."""

        # Create directories
        snippets_dir = temp_dir / "snippets" / "data"
        snippets_dir.mkdir(parents=True, exist_ok=True)

        # Create mint.json content
        mint_json_content = {
            "name": "AG2",
            "logo": {"dark": "/logo/ag2-white.svg", "light": "/logo/ag2.svg"},
            "navigation": [
                {"group": "", "pages": ["docs/Home", "docs/Getting-Started"]},
                {
                    "group": "Installation",
                    "pages": [
                        "docs/installation/Installation",
                        "docs/installation/Docker",
                        "docs/installation/Optional-Dependencies",
                    ],
                },
                {"group": "API Reference", "pages": ["PLACEHOLDER"]},
                # {
                #     "group": "AutoGen Studio",
                #     "pages": [
                #         "docs/autogen-studio/getting-started",
                #         "docs/autogen-studio/usage",
                #         "docs/autogen-studio/faqs",
                #     ],
                # },
            ],
        }

        # Create NotebooksMetadata.mdx content
        notebooks_metadata_content = """{/*
    Auto-generated file - DO NOT EDIT
    Please edit the add_front_matter_to_metadata_mdx function in process_notebooks.py
    */}

    export const notebooksMetadata = [
        {
            "title": "Using RetrieveChat Powered by MongoDB Atlas for Retrieve Augmented Code Generation and Question Answering",
            "link": "/notebooks/agentchat_RetrieveChat_mongodb",
            "description": "Explore the use of RetrieveChat for tasks like code generation from docstrings, answering complex questions with human feedback, and exploiting features like Update Context, custom prompts, and few-shot learning.",
            "image": null,
            "tags": [
                "MongoDB",
                "integration",
                "RAG"
            ],
            "source": "/notebook/agentchat_RetrieveChat_mongodb.ipynb"
        },
        {
            "title": "Mitigating Prompt hacking with JSON Mode",
            "link": "/notebooks/JSON_mode_example",
            "description": "Use JSON mode and Agent Descriptions to mitigate prompt manipulation and control speaker transition.",
            "image": null,
            "tags": [
                "JSON",
                "description",
                "prompt hacking",
                "group chat",
                "orchestration"
            ],
            "source": "/notebook/JSON_mode_example.ipynb"
        }];"""

        # Write files
        mint_json_path = temp_dir / "mint.json"
        with open(mint_json_path, "w", encoding="utf-8") as f:
            json.dump(mint_json_content, f, indent=2)
            f.write("\n")

        metadata_path = snippets_dir / "NotebooksMetadata.mdx"
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(notebooks_metadata_content)

    def test_extract_example_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self.setup(tmp_path)

            # Run the function
            metadata_path = tmp_path / "snippets" / "data" / "NotebooksMetadata.mdx"
            actual = extract_example_group(metadata_path)

            expected = {
                "group": "Examples",
                "pages": [
                    {
                        "group": "Examples by Notebook",
                        "pages": [
                            "notebooks/Notebooks",
                            "notebooks/agentchat_RetrieveChat_mongodb",
                            "notebooks/JSON_mode_example",
                        ],
                    },
                    "notebooks/Gallery",
                ],
            }

            assert actual == expected, actual


class TestAddAuthorsAndSocialImgToBlogPosts:
    @pytest.fixture
    def test_dir(self) -> Generator[Path, None, None]:
        """Create temporary test directory with blog posts and authors file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            website_dir = Path(tmp_dir)
            blog_dir = website_dir / "_blogs"
            blog_dir.mkdir()

            # Create first blog post
            post1_dir = blog_dir / "2023-04-21-LLM-tuning-math"
            post1_dir.mkdir()
            post1_content = textwrap.dedent("""
                ---
                title: Does Model and Inference Parameter Matter in LLM Applications? - A Case Study for MATH
                authors: sonichi
                tags: [LLM, GPT, research]
                ---

                lorem ipsum""").lstrip()
            (post1_dir / "index.mdx").write_text(post1_content)

            # Create second blog post
            post2_dir = blog_dir / "2023-06-28-MathChat"
            post2_dir.mkdir()
            post2_content = textwrap.dedent("""
                ---
                title: Introducing RealtimeAgent Capabilities in AG2
                authors:
                    - marklysze
                    - sternakt
                    - davorrunje
                    - davorinrusevljan
                tags: [Realtime API, Voice Agents, Swarm Teams, Twilio, AI Tools]
                ---

                <div>
                    <img noZoom className="social-share-img"
                    src="https://media.githubusercontent.com/media/ag2ai/ag2/refs/heads/main/website/static/img/cover.png"
                    alt="social preview"
                    style={{ position: 'absolute', left: '-9999px' }}
                    />
                </div>

                <div class="blog-authors">
                    <p class="authors">Authors:</p>
                    <CardGroup cols={2}>
                        <Card href="https://github.com/marklysze">
                            <div class="col card">
                            <div class="img-placeholder">
                                <img noZoom src="https://github.com/marklysze.png" />
                            </div>
                            <div>
                                <p class="name">Mark Sze</p>
                                <p>Software Engineer at AG2.ai</p>
                            </div>
                            </div>
                        </Card>
                        <Card href="https://github.com/sternakt">
                            <div class="col card">
                            <div class="img-placeholder">
                                <img noZoom src="https://github.com/sternakt.png" />
                            </div>
                            <div>
                                <p class="name">Tvrtko Sternak</p>
                                <p>Machine Learning Engineer at Airt</p>
                            </div>
                            </div>
                        </Card>
                        <Card href="https://github.com/davorrunje">
                            <div class="col card">
                            <div class="img-placeholder">
                                <img noZoom src="https://github.com/davorrunje.png" />
                            </div>
                            <div>
                                <p class="name">Davor Runje</p>
                                <p>CTO at Airt</p>
                            </div>
                            </div>
                        </Card>
                        <Card href="https://github.com/davorinrusevljan">
                            <div class="col card">
                            <div class="img-placeholder">
                                <img noZoom src="https://github.com/davorinrusevljan.png" />
                            </div>
                            <div>
                                <p class="name">Davorin</p>
                                <p>Developer</p>
                            </div>
                            </div>
                        </Card>
                    </CardGroup>
                </div>

                lorem ipsum""").lstrip()
            (post2_dir / "index.mdx").write_text(post2_content)

            # Create authors.yml
            authors_content = textwrap.dedent("""
                sonichi:
                    name: Chi Wang
                    title: Founder of AutoGen (now AG2) & FLAML
                    url: https://www.linkedin.com/in/chi-wang-autogen/
                    image_url: https://github.com/sonichi.png

                marklysze:
                    name: Mark Sze
                    title: Software Engineer at AG2.ai
                    url: https://github.com/marklysze
                    image_url: https://github.com/marklysze.png

                sternakt:
                    name: Tvrtko Sternak
                    title: Machine Learning Engineer at Airt
                    url: https://github.com/sternakt
                    image_url: https://github.com/sternakt.png

                davorrunje:
                    name: Davor Runje
                    title: CTO at Airt
                    url: https://github.com/davorrunje
                    image_url: https://github.com/davorrunje.png

                davorinrusevljan:
                    name: Davorin
                    title: Developer
                    url: https://github.com/davorinrusevljan
                    image_url: https://github.com/davorinrusevljan.png
                """).lstrip()
            (blog_dir / "authors.yml").write_text(authors_content)

            yield website_dir

    def test_add_authors_and_social_img(self, test_dir: Path) -> None:
        # Run the function
        add_authors_and_social_img_to_blog_posts(test_dir)

        # Get directory paths
        generated_blog_dir = test_dir / "blog"
        blog_dir = test_dir / "_blogs"

        # Verify directory structure matches
        blog_files = set(p.relative_to(blog_dir) for p in blog_dir.glob("**/*.mdx"))
        generated_files = set(p.relative_to(generated_blog_dir) for p in generated_blog_dir.glob("**/*.mdx"))
        assert blog_files == generated_files

        # Verify number of files matches
        assert len(list(blog_dir.glob("**/*.mdx"))) == len(list(generated_blog_dir.glob("**/*.mdx")))

        # Verify content of first blog post
        post1_path = generated_blog_dir / "2023-04-21-LLM-tuning-math" / "index.mdx"
        actual = post1_path.read_text()
        assert '<img noZoom className="social-share-img"' in actual
        assert '<p class="name">Chi Wang</p>' in actual
        assert '<p class="name">Davor Runje</p>' not in actual

        assert actual.count('<div class="blog-authors">') == 1
        assert actual.count('<p class="name">Chi Wang</p>') == 1

        # Verify content of second blog post
        post2_path = generated_blog_dir / "2023-06-28-MathChat" / "index.mdx"
        actual = post2_path.read_text()

        assert '<img noZoom className="social-share-img"' in actual
        assert '<p class="name">Mark Sze</p>' in actual
        assert '<p class="name">Tvrtko Sternak</p>' in actual
        assert '<p class="name">Davor Runje</p>' in actual
        assert '<p class="name">Davorin</p>' in actual
        assert '<p class="name">Chi Wang</p>' not in actual

        assert actual.count('<div class="blog-authors">') == 1
        assert actual.count('<p class="name">Mark Sze</p>') == 1


class TestConvertCalloutBlocks:
    @pytest.fixture
    def content(self) -> str:
        return textwrap.dedent("""
            # Title

            ## Introduction

            This is an introduction.

            ## Callout

            :::note
            This is a note.
            :::

            :::warning
            This is a warning.
            :::

            :::tip
            This is a tip.
            :::

            :::info
            This is an info.
            :::

            ````{=mdx}
            :::tabs
            <Tab title="OpenAI">
                - `model` (str, required): The identifier of the model to be used, such as 'gpt-4', 'gpt-3.5-turbo'.
            </Tab>
            <Tab title="Azure OpenAI">
                - `model` (str, required): The deployment to be used. The model corresponds to the deployment name on Azure OpenAI.
            </Tab>
            <Tab title="Other OpenAI compatible">
                - `model` (str, required): The identifier of the model to be used, such as 'llama-7B'.
            </Tab>
            :::
            ````

            ## Conclusion

            This is a conclusion.
            """)

    @pytest.fixture
    def expected(self) -> str:
        return textwrap.dedent("""
            # Title

            ## Introduction

            This is an introduction.

            ## Callout


            <div class="note">
            <Note>
            This is a note.
            </Note>
            </div>


            <div class="warning">
            <Warning>
            This is a warning.
            </Warning>
            </div>


            <div class="tip">
            <Tip>
            This is a tip.
            </Tip>
            </div>


            <div class="info">
            <Info>
            This is an info.
            </Info>
            </div>


            <div class="tabs">
            <Tabs>
            <Tab title="OpenAI">
                - `model` (str, required): The identifier of the model to be used, such as 'gpt-4', 'gpt-3.5-turbo'.
            </Tab>
            <Tab title="Azure OpenAI">
                - `model` (str, required): The deployment to be used. The model corresponds to the deployment name on Azure OpenAI.
            </Tab>
            <Tab title="Other OpenAI compatible">
                - `model` (str, required): The identifier of the model to be used, such as 'llama-7B'.
            </Tab>
            </Tabs>
            </div>


            ## Conclusion

            This is a conclusion.
            """)

    def test_convert_callout_blocks(self, content: str, expected: str) -> None:
        actual = convert_callout_blocks(content)
        assert actual == expected, actual
