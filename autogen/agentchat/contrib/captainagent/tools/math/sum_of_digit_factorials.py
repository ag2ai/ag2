# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
def sum_of_digit_factorials(number):
    """Calculates the sum of the factorial of each digit in a number, often used in problems involving curious numbers like 145.

    Args:
        number (int): The number for which to calculate the sum of digit factorials.

    Returns:
        int: The sum of the factorials of the digits in the given number.
    """
    from math import factorial

    return sum(factorial(int(digit)) for digit in str(number))
