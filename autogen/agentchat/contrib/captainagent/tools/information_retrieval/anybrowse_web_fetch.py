# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
def anybrowse_web_fetch(url, api_key=None):
    """Fetch a web page and convert it to clean Markdown using anybrowse.

    Handles JavaScript-heavy sites, Cloudflare-protected pages, and PDFs
    by rendering with a real Chrome browser. Returns LLM-ready Markdown.

    Free tier: 10 calls/day without signup, 50 calls/day with free account.
    Get an API key at https://anybrowse.dev/signup.

    Args:
        url (str): The URL to fetch and convert to Markdown.
        api_key (str, optional): anybrowse API key. If not provided,
            reads from ANYBROWSE_API_KEY environment variable. Not required
            for the free tier (10 calls/day).

    Returns:
        str: The page content as clean Markdown, or an error message.
    """
    import json
    import os

    import requests

    api_key = api_key or os.getenv("ANYBROWSE_API_KEY")

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"url": url}

    try:
        response = requests.post(
            "https://anybrowse.dev/scrape",
            headers=headers,
            data=json.dumps(payload),
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()

        if result.get("status") == "ok" and result.get("markdown"):
            title = result.get("title", url)
            return f"# {title}\n\n{result['markdown']}"
        else:
            return f"Error: {result.get('error', 'Unknown error from anybrowse')}"

    except requests.exceptions.Timeout:
        return f"Error: Request timed out while fetching {url}"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return "Error: Rate limit exceeded. Sign up at https://anybrowse.dev/signup for more calls."
        return f"Error: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error fetching {url}: {e!s}"
