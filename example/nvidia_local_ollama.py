# Example: NVIDIA hosted model with AG2's OpenAI compatible client
# pip install ag2[ollama]

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.llm_config.config import LLMConfig

# Use Ollama client with locally run NVIDIA model
llm_config = LLMConfig(
    {
        "api_type": "ollama",
        "model": "nemotron-3-nano",
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

# OUTPUT SAMPLE:

# user (to python_agent):

# Write a Python function that checks if a number is prime.

# --------------------------------------------------------------------------------

# >>>>>>>> USING AUTO REPLY...
# python_agent (to user):

# Here’s a compact, well‑documented implementation that works for any non‑negative integer `n`.
# It returns **True** if `n` is a prime number and **False** otherwise.

# ```python
# def is_prime(n: int) -> bool:
#     """
#     Return True if n is a prime number, False otherwise.

#     A prime is an integer greater than 1 that has no positive divisors
#     other than 1 and itself.

#     Parameters
#     ----------
#     n : int
#         The number to test.  Negative numbers, 0 and 1 are not prime.

#     Returns
#     -------
#     bool
#         True if n is prime, False otherwise.

#     Examples
#     --------
#     >>> is_prime(2)
#     True
#     >>> is_prime(37)
#     True
#     >>> is_prime(100)
#     False
#     """
#     # 1. Handle trivial cases up front – they are not prime.
#     if n <= 1:
#         return False                # 0, 1 and all negatives are non‑prime
#     if n == 2 or n == 3:
#         return True                 # the two smallest primes

#     # 2. Eliminate even numbers and multiples of 3 immediately.
#     if n % 2 == 0 or n % 3 == 0:
#         return False

#     # 3. Test only odd divisors up to sqrt(n).
#     #    Every composite > 3 can be written as (6k ± 1), so we step by 6
#     #    and check both candidates.
#     i = 5
#     while i * i <= n:
#         if n % i == 0 or n % (i + 2) == 0:
#             return False           # found a divisor → not prime
#         i += 6

#     # If we never returned, no divisor was found ⇒ n is prime.
#     return True


# # ----------------------------------------------------------------------
# # Quick sanity test (run only when this file is executed directly)
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     test_vals = [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 17,
#                  25, 29, 31, 37, 97, 100, -3]
#     for v in test_vals:
#         print(f"{v:>4} → {is_prime(v)}")
# ```

# ### How it works
# 1. **Trivial rejects** – numbers ≤ 1 are never prime; `2` and `3` are the smallest primes.
# 2. **Fast elimination** – if `n` is even (`% 2 == 0`) or a multiple of three, we can bail out immediately.
# 3. **Trial division up to √n** – any non‑trivial divisor must appear by √n.
#    Instead of testing every integer, we use the fact that all primes > 3 are of the form `6k ± 1`. We therefore check divisibility only for numbers like `5, 7, 11, 13, …` (step = 6) and their companions (`i + 2`).
#    This cuts the loop roughly to **O(√n / 6)** operations.

# ### Complexity
# * **Time:** O(√n) in the worst case, but with a small constant factor thanks to skipping even numbers and multiples of three.
# * **Space:** O(1) – only a few integer variables are used.

# Feel free to copy this function into your own projects or adapt it (e.g., raise `ValueError` for non‑integers, handle big integers via probabilistic tests like Miller‑Rabin if needed). Happy coding!

# --------------------------------------------------------------------------------

# >>>>>>>> TERMINATING RUN (747dcc4f-8bb4-4eea-bdcb-9087fc8a0f48): Maximum turns (1) reached
