# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
# alternative api: https://rapidapi.com/omarmhaimdat/api/youtube-v2


from typing import Any


def get_youtube_caption(video_id: str) -> Any:
    """Retrieves the captions for a YouTube video.

    Args:
        videoId (str): The ID of the YouTube video.

    Returns:
        str: The captions of the YouTube video in text format.

    Raises:
        KeyError: If the RAPID_API_KEY environment variable is not set.
    """
    import os

    import requests

    rapid_api_key = os.environ["RAPID_API_KEY"]
    video_url = f"https://www.youtube.com/watch?v={video_id: str}"
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"

    querystring = {"url": video_url, "lang": "en", "flat_text": "true"}

    headers = {"X-RapidAPI-Key": rapid_api_key, "X-RapidAPI-Host": "youtube-transcript3.p.rapidapi.com"}

    response = requests.get(url, headers=headers, params=querystring)
    response = response.json()
    print(response)
    return response["transcript"]
