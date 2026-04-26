# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.tools.builtin.web_search import UserLocation


class TestUserLocationEquality:
    def test_equal_full(self) -> None:
        loc1 = UserLocation(city="Berlin", region="Berlin", country="DE", timezone="Europe/Berlin")
        loc2 = UserLocation(city="Berlin", region="Berlin", country="DE", timezone="Europe/Berlin")
        assert loc1 == loc2

    def test_not_equal_different_country(self) -> None:
        loc1 = UserLocation(country="DE", timezone="Europe/Berlin")
        loc2 = UserLocation(country="US", timezone="America/New_York")
        assert loc1 != loc2

    def test_equal_partial_vs_full_same_specified_fields(self) -> None:
        # When one location specifies only country and the other specifies
        # country + timezone, they are equal because timezone is None in the first.
        loc1 = UserLocation(country="DE")
        loc2 = UserLocation(country="DE", timezone="Europe/Berlin")
        assert loc1 == loc2

    def test_not_equal_empty_vs_specified(self) -> None:
        # An empty location (all None) vs a specified location - should be equal
        # since no fields are specified in either.
        loc1 = UserLocation()
        loc2 = UserLocation(country="DE")
        assert loc1 == loc2

    def test_equal_both_empty(self) -> None:
        loc1 = UserLocation()
        loc2 = UserLocation()
        assert loc1 == loc1

    def test_not_equal_same_city_different_country(self) -> None:
        loc1 = UserLocation(city="Berlin", country="DE")
        loc2 = UserLocation(city="Berlin", country="US")
        assert loc1 != loc2

    def test_equal_only_country_both(self) -> None:
        loc1 = UserLocation(country="DE")
        loc2 = UserLocation(country="DE")
        assert loc1 == loc2

    def test_not_equal_same_country_different_city(self) -> None:
        loc1 = UserLocation(city="Berlin", country="DE")
        loc2 = UserLocation(city="Munich", country="DE")
        assert loc1 != loc2

    def test_not_equal_with_none_in_both(self) -> None:
        loc1 = UserLocation(country="DE", city=None)
        loc2 = UserLocation(country="DE")
        # Both have country="DE", city is None in loc1 and not specified in loc2
        # This should be equal since None city means "not specified"
        assert loc1 == loc2
