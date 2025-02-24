# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import textwrap
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union

import pytest

from autogen._website.process_notebooks import (
    add_authors_and_social_img_to_blog_and_user_stories,
    add_front_matter_to_metadata_mdx,
    cleanup_tmp_dirs,
    convert_callout_blocks,
    copy_images_from_notebooks_dir_to_target_dir,
    ensure_mint_json_exists,
    extract_example_group,
    extract_img_tag_from_figure_tag,
    generate_nav_group,
    get_files_path_from_navigation,
    get_sorted_files,
    update_group_pages,
)
from autogen._website.utils import NavigationGroup
from autogen.import_utils import skip_on_missing_imports


class TestUpdateGroupPages:
    @pytest.fixture
    def sample_navigation(self) -> list[dict[str, Union[str, list[Union[str, dict[str, Union[str, list[str]]]]]]]]:
        return [
            {"group": "Home", "pages": ["docs/home/Home"]},
            {
                "group": "Use Cases",
                "pages": [
                    {
                        "group": "Use cases",
                        "pages": [
                            "docs/use-cases/use-cases/customer-service",
                        ],
                    },
                    {"group": "Notebooks", "pages": ["docs/use-cases/notebooks/notebooks"]},
                    {
                        "group": "Community Gallery",
                        "pages": ["docs/use-cases/community-gallery/community-gallery"],
                    },
                ],
            },
            {"group": "FAQs", "pages": ["faq/FAQ"]},
        ]

    def test_update_top_level_group(
        self, sample_navigation: list[dict[str, Union[str, list[Union[str, dict[str, Union[str, list[str]]]]]]]]
    ) -> None:
        updated_pages = ["docs/use-cases/use-cases/customer-service"]
        target_grp = "Use Cases"
        updated_navigation = update_group_pages(sample_navigation, target_grp, updated_pages)
        expected_navigation = [
            {"group": "Home", "pages": ["docs/home/Home"]},
            {
                "group": "Use Cases",
                "pages": ["docs/use-cases/use-cases/customer-service"],
            },
            {"group": "FAQs", "pages": ["faq/FAQ"]},
        ]

        assert updated_navigation == expected_navigation, updated_navigation
        assert sample_navigation != updated_navigation

    def test_update_nested_group(
        self, sample_navigation: list[dict[str, Union[str, list[Union[str, dict[str, Union[str, list[str]]]]]]]]
    ) -> None:
        updated_pages = ["docs/use-cases/updated-notebook/index"]
        target_grp = "Notebooks"
        updated_navigation = update_group_pages(sample_navigation, target_grp, updated_pages)
        expected_navigation = [
            {"group": "Home", "pages": ["docs/home/Home"]},
            {
                "group": "Use Cases",
                "pages": [
                    {
                        "group": "Use cases",
                        "pages": [
                            "docs/use-cases/use-cases/customer-service",
                        ],
                    },
                    {"group": "Notebooks", "pages": ["docs/use-cases/updated-notebook/index"]},
                    {
                        "group": "Community Gallery",
                        "pages": ["docs/use-cases/community-gallery/community-gallery"],
                    },
                ],
            },
            {"group": "FAQs", "pages": ["faq/FAQ"]},
        ]

        assert updated_navigation == expected_navigation, updated_navigation
        assert sample_navigation != updated_navigation


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
        notebooks_dir = tmp_path / "docs" / "use-cases" / "notebooks" / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        (notebooks_dir / "example-1.mdx").touch()
        (notebooks_dir / "example-2.mdx").touch()

        cleanup_tmp_dirs(tmp_path, False)
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

        cleanup_tmp_dirs(tmp_path, False)
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

            with open(metadata_dir / "NotebooksMetadata.mdx") as f:
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
        "link": "/docs/use-cases/notebooks/notebooks/source",
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
            ), actual

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

            with open(metadata_dir / "NotebooksMetadata.mdx") as f:
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
        "link": "/docs/use-cases/notebooks/notebooks/source",
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

            expected = [
                "docs/use-cases/notebooks/Notebooks",
                "docs/use-cases/notebooks/notebooks/agentchat_RetrieveChat_mongodb",
                "docs/use-cases/notebooks/notebooks/JSON_mode_example",
            ]

            assert actual == expected, actual


class TestAddAuthorsAndSocialImgToBlogPosts:
    @pytest.fixture
    def test_dir(self) -> Generator[Path, None, None]:
        """Create temporary test directory with blog posts and authors file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            website_dir = Path(tmp_dir)
            blog_dir = website_dir / "docs" / "_blogs"
            blog_dir.mkdir(parents=True)

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

            # Create blogs_and_user_stories_authors.yml
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
            (website_dir / "blogs_and_user_stories_authors.yml").write_text(authors_content)

            yield website_dir

    @skip_on_missing_imports("yaml", "docs")
    def test_add_authors_and_social_img(self, test_dir: Path) -> None:
        # Run the function
        add_authors_and_social_img_to_blog_and_user_stories(test_dir)

        # Get directory paths
        generated_blog_dir = test_dir / "docs" / "blog"
        blog_dir = test_dir / "docs" / "_blogs"

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


class TestEditLinks:
    @pytest.fixture
    def navigation(self) -> list[NavigationGroup]:
        return [
            {"group": "Home", "pages": ["docs/home/home"]},
            {
                "group": "User Guide",
                "pages": [
                    {
                        "group": "Basic Concepts",
                        "pages": [
                            "docs/user-guide/basic-concepts/installing-ag2",
                            "docs/user-guide/basic-concepts/llm-configuration",
                        ],
                    },
                    {"group": "Advanced Concepts", "pages": ["docs/user-guide/advanced-concepts/rag"]},
                    {"group": "Model Providers", "pages": ["docs/user-guide/models/openai"]},
                    {"group": "Reference Agents", "pages": ["docs/user-guide/reference-agents/index"]},
                ],
            },
            {
                "group": "Use Cases",
                "pages": [
                    {"group": "Use cases", "pages": ["docs/use-cases/use-cases/customer-service"]},
                    {"group": "Notebooks", "pages": ["docs/use-cases/notebooks/notebooks"]},
                    "docs/use-cases/community-gallery/community-gallery",
                ],
            },
            {"group": "Contributor Guide", "pages": ["docs/contributor-guide/contributing"]},
            {"group": "FAQs", "pages": ["docs/faq/FAQ"]},
            {"group": "Ecosystem", "pages": ["docs/ecosystem/agentops"]},
        ]

    def test_get_files_path_from_navigation(self, navigation: list[NavigationGroup]) -> None:
        expected_files = [
            "docs/home/home",
            "docs/user-guide/basic-concepts/installing-ag2",
            "docs/user-guide/basic-concepts/llm-configuration",
            "docs/user-guide/advanced-concepts/rag",
            "docs/user-guide/models/openai",
            "docs/user-guide/reference-agents/index",
            "docs/use-cases/use-cases/customer-service",
            "docs/use-cases/notebooks/notebooks",
            "docs/use-cases/community-gallery/community-gallery",
            "docs/contributor-guide/contributing",
            "docs/faq/FAQ",
            "docs/ecosystem/agentops",
        ]

        expected = [Path(f) for f in expected_files]

        actual = get_files_path_from_navigation(navigation)
        assert actual == expected, actual


class TestExtractImgTagFromFigureTag:
    @pytest.fixture
    def content(self) -> str:
        return textwrap.dedent("""
            # Title

            ## Introduction

            This is an introduction.

            ## Figure

            <figure>
                <img src="https://example.com/image.png" alt="image" />
                <figcaption>Image caption</figcaption>
            </figure>

            <figure>
                <img src="swarm_enhanced_01.png" alt="image" />
                <figcaption>Image caption</figcaption>
            </figure>

            <figure>
            <img
            src="https://media.githubusercontent.com/media/ag2ai/ag2/main/notebook/friendly_and_suspicous.jpg"
            alt="agent flow" />
            <figcaption aria-hidden="true">agent flow</figcaption>
            </figure>

            ## Conclusion

            This is a conclusion.
            """).lstrip()

    @pytest.fixture
    def expected(self) -> str:
        return textwrap.dedent("""
            # Title

            ## Introduction

            This is an introduction.

            ## Figure

            <img src="https://example.com/image.png" alt="image" />

            <img src="/docs/use-cases/notebooks/notebooks/swarm_enhanced_01.png" alt="image" />

            <img
            src="https://media.githubusercontent.com/media/ag2ai/ag2/main/notebook/friendly_and_suspicous.jpg"
            alt="agent flow" />

            ## Conclusion

            This is a conclusion.
            """).lstrip()

    def test_extract_img_tag_from_figure_tag(self, content: str, expected: str) -> None:
        img_rel_path = Path("docs/use-cases/notebooks/notebooks")
        actual = extract_img_tag_from_figure_tag(content, img_rel_path)
        assert actual == expected, actual


class TestCopyImagesFromNotebooksDirToTargetDir:
    @pytest.fixture
    def setup_test_files(self, tmp_path: Path) -> dict[str, Path]:
        # Create directories
        notebook_dir = tmp_path / "notebooks"
        notebook_sub_dir = notebook_dir / "sub_dir"
        notebook_dir.mkdir()
        notebook_sub_dir.mkdir()

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        # Create test files
        files_to_create = [
            (notebook_dir / "test1.png", b"png content"),
            (notebook_dir / "test2.jpg", b"jpg content"),
            (notebook_dir / "test3.txt", b"text content"),
            (notebook_sub_dir / "test1.png", b"png content"),
            (notebook_sub_dir / "test2.jpg", b"jpg content"),
        ]

        for file_path, content in files_to_create:
            file_path.write_bytes(content)

        return {"notebook_dir": notebook_dir, "target_dir": target_dir}

    def test_copy_images(self, setup_test_files: dict[str, Path]) -> None:
        copy_images_from_notebooks_dir_to_target_dir(setup_test_files["notebook_dir"], setup_test_files["target_dir"])

        # Check only images were copied
        copied_files = list(setup_test_files["target_dir"].iterdir())
        assert len(copied_files) == 2
        assert (setup_test_files["target_dir"] / "test1.png").exists()
        assert (setup_test_files["target_dir"] / "test2.jpg").exists()
        assert not (setup_test_files["target_dir"] / "test3.txt").exists()
        assert not (setup_test_files["target_dir"] / "sub_dir").exists()
