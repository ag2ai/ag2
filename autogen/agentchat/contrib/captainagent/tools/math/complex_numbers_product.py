# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
from autogen.coding.func_with_reqs import with_requirements


@with_requirements(["sympy"])
def complex_numbers_product(complex_numbers):
    """Calculates the product of a list of complex numbers.

    Args:
        complex_numbers (list): A list of dictionaries representing complex numbers.
            Each dictionary should have 'real' and 'imag' keys representing the real
            and imaginary parts of the complex number.

    Returns:
        complex: The simplified product of the complex numbers.

    """
    from sympy import I, simplify

    result = 1
    for c in complex_numbers:
        result *= c["real"] + I * c["imag"]
    return simplify(result)
