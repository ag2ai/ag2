# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
# alternative api: https://rapidapi.com/omarmhaimdat/api/youtube-v2


def get_youtube_caption(videoId):
    """
    Retrieves the captions for a YouTube video.

    Args:
        videoId (str): The ID of the YouTube video.

    Returns:
        str: The captions of the YouTube video in text format.

    Raises:
        KeyError: If the RAPID_API_KEY environment variable is not set.
    """
    import os

    import requests

    RAPID_API_KEY = os.environ["RAPID_API_KEY"]
    video_url = f"https://www.youtube.com/watch?v={videoId}"
    url = "https://youtube-transcript3.p.rapidapi.com/api/transcript-with-url"

    querystring = {"url": video_url, "lang": "en", "flat_text": "true"}

    headers = {"X-RapidAPI-Key": RAPID_API_KEY, "X-RapidAPI-Host": "youtube-transcript3.p.rapidapi.com"}

    response = requests.get(url, headers=headers, params=querystring)
    response = response.json()
    print(response)
    return response["transcript"]
