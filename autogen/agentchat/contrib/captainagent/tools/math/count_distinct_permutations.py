# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
def count_distinct_permutations(sequence):
    """Counts the number of distinct permutations of a sequence where items may be indistinguishable.

    Args:
        sequence (iterable): The sequence for which to count the distinct permutations.

    Returns:
        int: The number of distinct permutations.

    Example:
        >>> count_distinct_permutations("aab")
        3
        >>> count_distinct_permutations([1, 2, 2])
        3
    """
    from collections import Counter
    from math import factorial

    counts = Counter(sequence)
    total_length = sum(counts.values())
    permutations = factorial(total_length)
    for count in counts.values():
        permutations //= factorial(count)
    return permutations
