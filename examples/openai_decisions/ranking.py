"""Ranking: items need to be ordered by semantic relevance or quality.

Search results, recommendations, candidate prioritization. Each ``ask()`` is one
question, so ranking is one ``predicate`` per candidate, asked in parallel, and a
sort on the probabilities. Search and retrieval are the same loop over a candidate
pool: keep the top results, or hand them to a generative agent as context.
Needs ``OPENAI_API_KEY``.
"""

import asyncio

from ag2 import Agent, ResponseSchema
from ag2.config import OpenAIDecisionsConfig
from ag2.events import DataInput

QUERY = "How do I rotate an API key without downtime?"

DOCUMENTS = [
    "API keys are rotated from Settings > API. Create the new key first, deploy it, then revoke the old one.",
    "Rate limits are applied per API key: 600 requests per minute on the Team plan.",
    "Webhooks retry failed deliveries with exponential backoff for up to 24 hours.",
    "Revoking a key takes effect immediately; requests using it fail with 401 from then on.",
]


async def main() -> None:
    judge = Agent(
        "relevance",
        config=OpenAIDecisionsConfig(),
        response_schema=ResponseSchema(bool, description="The document answers the query."),
    )

    # Structured data is sent as JSON text, so the query and the candidate stay labelled.
    replies = await asyncio.gather(*(judge.ask(DataInput({"query": QUERY, "document": doc})) for doc in DOCUMENTS))

    scored = [(reply.response.metadata["probability"], document) for reply, document in zip(replies, DOCUMENTS)]
    for probability, document in sorted(scored, reverse=True):
        print(f"{probability:.2f}  {document}")


if __name__ == "__main__":
    asyncio.run(main())
