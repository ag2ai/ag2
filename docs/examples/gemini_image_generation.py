"""Example: Generate or edit an image with a Gemini agent using the IMAGE response modality.

Gemini produces images by selecting an image-capable model and requesting the
``IMAGE`` response modality (no tool needed). Pass an existing image with
``ImageInput`` and the ``--edit`` flag to modify it instead of generating from
scratch. Generated images come back as ``BinaryResult`` objects on ``reply.files``.

Requires:
    - GOOGLE_API_KEY or GEMINI_API_KEY set in environment (or .env)
    - pip install google-genai

Usage:
    # Text -> image (generation)
    python docs/examples/gemini_image_generation.py
    python docs/examples/gemini_image_generation.py "a watercolor fox in a snowy forest"

    # Image + text -> image (editing)
    python docs/examples/gemini_image_generation.py "put a party hat on the robot" --edit gemini_image_0.jpeg
"""

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from autogen.beta import Agent
from autogen.beta.config import GeminiConfig
from autogen.beta.events import ImageInput

load_dotenv()  # Load environment variables from .env file, if it exists

# Generated images are written here. Defaults to the repository root so the files
# are easy to find regardless of the current working directory.
OUTPUT_DIR = Path(__file__).resolve().parents[2]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or edit an image with Gemini.")
    parser.add_argument("prompt", nargs="?", help="What to generate, or how to edit the --edit image.")
    parser.add_argument("--edit", metavar="IMAGE", help="Path to an image to edit instead of generating from scratch.")
    args = parser.parse_args()

    agent = Agent(
        "image_agent",
        "You generate and edit images when asked.",
        config=GeminiConfig(
            model="gemini-3.1-flash-image",
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    if args.edit:
        prompt = args.prompt or "put a fun party hat on the subject"
        # Pass the source image alongside the instruction to edit it in place.
        reply = await agent.ask(f"{prompt}. Keep everything else the same.", ImageInput(path=args.edit))
        stem = "gemini_edit"
    else:
        prompt = args.prompt or "a friendly robot waving hello, simple flat illustration"
        reply = await agent.ask(f"Generate an image of {prompt}.")
        stem = "gemini_image"

    if reply.body:
        print(reply.body)

    if not reply.files:
        print("No image was returned.")
        return

    for index, image in enumerate(reply.files):
        media_type = image.metadata.get("media_type", "image/png")
        extension = media_type.split("/")[-1]
        filename = OUTPUT_DIR / f"{stem}_{index}.{extension}"
        filename.write_bytes(image.data)
        print(f"Saved {filename} ({media_type}, {len(image.data)} bytes)")


if __name__ == "__main__":
    asyncio.run(main())
