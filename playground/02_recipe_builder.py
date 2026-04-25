"""02 · Recipe builder — tools and structured output

Shows two core Actor features on top of the bare loop:

1. A custom ``@tool`` function the LLM can call (``scale_ingredient``).
2. A Pydantic ``response_schema`` so the final reply is a typed object.

Run::

    .venv-beta/bin/python playground/02_recipe_builder.py
"""

import asyncio

from _config import default_config, section
from pydantic import BaseModel, Field

from autogen.beta import Actor


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
    config = default_config()

    section("Recipe builder — scale an existing dish for 6 servings")

    agent = Actor(
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
