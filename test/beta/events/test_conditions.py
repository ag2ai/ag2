from autogen.beta.events import BaseEvent


class TestEvent(BaseEvent):
    __test__ = False

    field: int | str | None


class AnotherEvent(BaseEvent):
    field: str


class TestEventConditions:
    def test_different_event_with_condition(self):
        condition = TestEvent.field == "1"

        assert condition(TestEvent(field="1"))
        assert not condition(AnotherEvent(field="1"))

    def test_equality_condition_string(self):
        condition = TestEvent.field == "1"

        assert condition(TestEvent(field="1"))
        assert not condition(TestEvent(field="2"))
        assert not condition(TestEvent(field=2))

    def test_equality_condition_integer(self):
        condition = TestEvent.field == 42

        assert condition(TestEvent(field=42))
        assert not condition(TestEvent(field=41))
        assert not condition(TestEvent(field="42"))

    def test_inequality_condition(self):
        condition = TestEvent.field != "1"

        assert not condition(TestEvent(field="1"))
        assert condition(TestEvent(field="2"))
        assert condition(TestEvent(field=2))

    def test_less_than_condition(self):
        condition = TestEvent.field < 10

        assert condition(TestEvent(field=5))
        assert condition(TestEvent(field=9))
        assert not condition(TestEvent(field=10))
        assert not condition(TestEvent(field=15))

    def test_less_than_or_equal_condition(self):
        condition = TestEvent.field <= 10

        assert condition(TestEvent(field=5))
        assert condition(TestEvent(field=10))
        assert not condition(TestEvent(field=11))

    def test_greater_than_condition(self):
        condition = TestEvent.field > 10

        assert not condition(TestEvent(field=10))
        assert condition(TestEvent(field=11))
        assert condition(TestEvent(field=20))

    def test_greater_than_or_equal_condition(self):
        condition = TestEvent.field >= 10

        assert not condition(TestEvent(field=9))
        assert condition(TestEvent(field=10))
        assert condition(TestEvent(field=20))

    def test_and_condition(self):
        condition = (TestEvent.field > 0) & (TestEvent.field < 10)

        assert condition(TestEvent(field=5))
        assert condition(TestEvent(field=1))
        assert condition(TestEvent(field=9))
        assert not condition(TestEvent(field=0))
        assert not condition(TestEvent(field=10))
        assert not condition(TestEvent(field=-5))
        assert not condition(TestEvent(field=15))

    def test_or_condition(self):
        condition = (TestEvent.field < 0) | (TestEvent.field > 10)

        assert condition(TestEvent(field=-5))
        assert condition(TestEvent(field=15))
        assert not condition(TestEvent(field=0))
        assert not condition(TestEvent(field=5))
        assert not condition(TestEvent(field=10))

    def test_or_condition_with_class(self):
        condition = (TestEvent.field > 10) | AnotherEvent | (TestEvent.field < 0)

        assert condition(TestEvent(field=15))
        assert condition(AnotherEvent(field=""))
        assert condition(TestEvent(field=-5))
        assert not condition(TestEvent(field=0))

    def test_or_condition_with_class_first(self):
        condition = AnotherEvent | (TestEvent.field > 10)

        assert condition(TestEvent(field=15))
        assert condition(AnotherEvent(field=""))
        assert not condition(TestEvent(field=0))

    def test_or_condition_with_class_or_method(self):
        condition = AnotherEvent.or_(TestEvent.field > 10)

        assert condition(TestEvent(field=15))
        assert condition(AnotherEvent(field=""))
        assert not condition(TestEvent(field=0))

    def test_or_condition_with_union_classes(self):
        condition = AnotherEvent | TestEvent

        assert condition(TestEvent(field=15))
        assert condition(AnotherEvent(field=""))

    def test_and_method(self):
        condition = (TestEvent.field > 0).and_(TestEvent.field < 10)

        assert condition(TestEvent(field=5))
        assert condition(TestEvent(field=1))
        assert condition(TestEvent(field=9))
        assert not condition(TestEvent(field=0))
        assert not condition(TestEvent(field=10))
        assert not condition(TestEvent(field=-5))
        assert not condition(TestEvent(field=15))

    def test_or_method(self):
        condition = (TestEvent.field < 0).or_(TestEvent.field > 10)

        assert condition(TestEvent(field=-5))
        assert condition(TestEvent(field=15))
        assert not condition(TestEvent(field=0))
        assert not condition(TestEvent(field=5))
        assert not condition(TestEvent(field=10))

    def test_not_condition(self):
        condition = ~(TestEvent.field == "1")

        assert not condition(TestEvent(field="1"))
        assert condition(TestEvent(field="2"))
        assert condition(TestEvent(field=2))

    def test_not_method(self):
        condition = (TestEvent.field == "1").not_()

        assert not condition(TestEvent(field="1"))
        assert condition(TestEvent(field="2"))
        assert condition(TestEvent(field=2))

    def test_complex_condition(self):
        condition = ((TestEvent.field >= 0) & (TestEvent.field <= 10)) | (TestEvent.field == 100)

        assert condition(TestEvent(field=0))
        assert condition(TestEvent(field=5))
        assert condition(TestEvent(field=10))
        assert condition(TestEvent(field=100))
        assert not condition(TestEvent(field=-1))
        assert not condition(TestEvent(field=11))
        assert not condition(TestEvent(field=99))

    def test_event_with_multiple_fields(self):
        event = TestEvent(field="test", value=42, name="example")

        assert event.field == "test"
        assert event.value == 42
        assert event.name == "example"

    def test_condition_with_none_value(self):
        condition = TestEvent.field == None  # noqa: E711

        assert condition(TestEvent(field=None))
        assert not condition(TestEvent(field="test"))
        assert not condition(TestEvent(field=0))

    def test_condition_with_is_none_value(self):
        condition = TestEvent.field.is_(None)

        assert condition(TestEvent(field=None))
        assert not condition(TestEvent(field="test"))
        assert not condition(TestEvent(field=0))

    def test_condition_with_boolean(self):
        condition = TestEvent.field == True  # noqa: E712

        assert condition(TestEvent(field=True))
        assert not condition(TestEvent(field=False))
        assert not condition(TestEvent(field=1))

    def test_chained_conditions(self):
        condition = (TestEvent.field > 0) & (TestEvent.field < 100) & (TestEvent.field != 50)

        assert condition(TestEvent(field=25))
        assert condition(TestEvent(field=75))
        assert not condition(TestEvent(field=0))
        assert not condition(TestEvent(field=100))
        assert not condition(TestEvent(field=50))
