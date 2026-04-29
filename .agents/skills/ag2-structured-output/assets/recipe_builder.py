"""Recipe builder — tools and structured output.

Mirrors website/docs/beta/code_examples/02_recipe_builder.mdx. Demonstrates
two core Agent features on top of the bare loop:

1. A custom @tool function the LLM can call (scale_ingredient).
2. A Pydantic response_schema so the final reply is a typed object.

Run::

    python recipe_builder.py
"""

import asyncio

from pydantic import BaseModel, Field

from autogen.beta import Agent
from autogen.beta.config import GeminiConfig


def section(title: str) -> None:
    print(f"\n── {title} ───")


class Ingredient(BaseModel):
    name: str
    quantity: float
    unit: str


class Recipe(BaseModel):
    title: str = Field(description="Short human title for the recipe.")
    servings: int = Field(description="How many portions this recipe yields.")
    ingredients: list[Ingredient]
    steps: list[str] = Field(description="Ordered preparation steps.")


def scale_ingredient(quantity: float, factor: float) -> float:
    """Return ``quantity`` multiplied by ``factor``, rounded to 2 decimals.

    The model uses this any time it needs to rescale a recipe for a
    different number of servings.
    """
    return round(quantity * factor, 2)


async def main() -> None:
    config = GeminiConfig(model="gemini-3-flash-preview", temperature=0)

    section("Recipe builder — scale an existing dish for 6 servings")

    agent = Agent(
        "chef",
        prompt=(
            "You are a culinary assistant. When asked to rescale a recipe, "
            "use the scale_ingredient tool for every ingredient to compute the "
            "new quantity. Return a complete Recipe object."
        ),
        config=config,
        tools=[scale_ingredient],
        response_schema=Recipe,
    )

    reply = await agent.ask(
        "Start from classic carbonara for 2 servings: 200g spaghetti, 2 eggs, "
        "100g guanciale, 50g pecorino romano. Rescale it for 6 servings and "
        "produce the full Recipe."
    )

    recipe: Recipe | None = await reply.content(retries=1)

    if recipe is None:
        print("Model returned no body — try again.")
        return

    print(f"{recipe.title}  ({recipe.servings} servings)")
    print()
    print("Ingredients:")
    for ing in recipe.ingredients:
        print(f"  - {ing.quantity} {ing.unit} {ing.name}")
    print()
    print("Steps:")
    for i, step in enumerate(recipe.steps, 1):
        print(f"  {i}. {step}")


if __name__ == "__main__":
    asyncio.run(main())
