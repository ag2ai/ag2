# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from contextlib import suppress


def modular_inverse_sum(expressions, modulus):
    """Calculates the sum of modular inverses of the given expressions modulo the specified modulus.

    Args:
        expressions (list): A list of numbers for which the modular inverses need to be calculated.
        modulus (int): The modulus value.

    Returns:
        int: The sum of modular inverses modulo the specified modulus.
    """
    from sympy import mod_inverse

    mod_sum = 0
    for number in expressions:
        with suppress(ValueError):
            mod_sum += mod_inverse(number, modulus)
    return mod_sum % modulus
