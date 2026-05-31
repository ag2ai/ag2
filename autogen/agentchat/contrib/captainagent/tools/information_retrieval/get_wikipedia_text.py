# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
def get_wikipedia_text(title):
    """Retrieves the text content of a Wikipedia page. It does not support tables and other complex formatting.

    Args:
        title (str): The title of the Wikipedia page.

    Returns:
        str or None: The text content of the Wikipedia page if it exists, None otherwise.
    """
    import wikipediaapi

    wiki_wiki = wikipediaapi.Wikipedia("Mozilla/5.0 (merlin@example.com)", "en")
    page = wiki_wiki.page(title)

    if page.exists():
        return page.text
    else:
        return None
