"""Detection: a yes/no question about an image.

A ``bool`` schema is a ``predicate`` question; the reply is ``True`` once the
probability reaches ``boolean_threshold``, and the raw probability is on the
reply's metadata. Images are sent inline as base64 data URLs, which is the only
image form the Decisions API accepts. Needs ``OPENAI_API_KEY``.

Usage: python -m examples.openai_decisions.damage_check product.png
"""

import asyncio
from pathlib import Path

from ag2 import Agent, ImageInput, ResponseSchema
from ag2.config import OpenAIDecisionsConfig


async def main(path: Path) -> None:
    inspector = Agent(
        "inspector",
        prompt="Ignore shadows and damage to the packaging.",
        config=OpenAIDecisionsConfig(
            boolean_threshold=0.8,
        ),
        response_schema=ResponseSchema(
            bool, description="Does the product have visible damage, such as a crack, tear, or dent?"
        ),
    )

    reply = await inspector.ask("Inspect the product in this photo.", ImageInput(path=path))

    damaged = await reply.content()
    print(f"path={path} damaged={damaged}  probability={reply.response.metadata['probability']:.2f}")


if __name__ == "__main__":
    asyncio.run(main(Path(__file__).parent / "damage_check_broken.png"))
    asyncio.run(main(Path(__file__).parent / "damage_check_ok.png"))
