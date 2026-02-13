# Example: NVIDIA hosted model with AG2's OpenAI compatible client
# Needs an NVIDIA API key from build.nvidia.com
# pip install ag2[openai]

import os
from pathlib import Path

from dotenv import load_dotenv

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.llm_config.config import LLMConfig

# NVIDIA_API_KEY stored in .env
load_dotenv(Path(__file__).parent / ".env")

# Use OpenAI client as NVIDIA endpoints are OpenAI API compatible
llm_config = LLMConfig(
    {
        "api_type": "openai",
        "model": "minimaxai/minimax-m2.1",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": os.environ["NVIDIA_API_KEY"],
    },
    temperature=1.0,
    top_p=0.95,
)

python_agent = ConversableAgent(
    name="python_agent",
    llm_config=llm_config,
)

response = python_agent.run(
    message="Write a Python function that checks if a number is prime.",
    max_turns=1,
)

response.process()

# SAMPLE OUTPUT:

# user (to python_agent):

# Write a Python function that checks if a number is prime.

# --------------------------------------------------------------------------------

# >>>>>>>> USING AUTO REPLY...
# [autogen.oai.client: 02-13 14:26:22] {738} WARNING - Model minimaxai/minimax-m2.1 is not found. The cost will be 0. In your config_list, add field {"price" : [prompt_price_per_1k, completion_token_price_per_1k]} for customized pricing.
# python_agent (to user):

# <think>We are going to write a function that checks if a number is prime.
#  A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.
#  We can do this by checking divisibility from 2 up to the square root of the number (inclusive).
# </think>

# Here's a Python function to check if a number is prime:

# ```python
# import math

# def is_prime(n):
#     """Check if a number is prime."""
#     if n <= 1:
#         return False
#     if n == 2:
#         return True
#     if n % 2 == 0:
#         return False

#     # Check odd divisors up to the square root of n
#     for i in range(3, int(math.sqrt(n)) + 1, 2):
#         if n % i == 0:
#             return False
#     return True
# ```

# **Key features:**
# 1. Handles edge cases: Numbers ≤1 are not prime, 2 is prime, and even numbers greater than 2 are not prime
# 2. Optimizes by:
#    - Checking divisibility only by odd numbers after 2
#    - Limiting checks to numbers ≤√n (since factors come in pairs)

# **Example usage:**
# ```python
# print(is_prime(7))    # True
# print(is_prime(4))    # False
# print(is_prime(2))    # True
# print(is_prime(1))    # False
# ```

# This implementation efficiently handles all cases while minimizing unnecessary computations.

# --------------------------------------------------------------------------------

# >>>>>>>> TERMINATING RUN (4e486c11-d7ef-4c11-b092-7416033f7963): Maximum turns (1) reached
