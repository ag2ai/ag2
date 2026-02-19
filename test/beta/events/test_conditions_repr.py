from autogen.beta.events import BaseEvent
from autogen.beta.events.conditions import TypeCondition


class TestEvent(BaseEvent):
    __test__ = False

    field: int


class AnotherEvent(BaseEvent):
    field: str


class TestOpConditionRepr:
    def test_equality_condition_repr(self):
        condition = TestEvent.field == 5
        assert repr(condition) == "Is(TestEvent.field == 5)"

    def test_equality_condition_repr_string(self):
        condition = TestEvent.field == "test"
        assert repr(condition) == "Is(TestEvent.field == 'test')"

    def test_inequality_condition_repr(self):
        condition = TestEvent.field != 5
        assert repr(condition) == "Is(TestEvent.field != 5)"

    def test_less_than_condition_repr(self):
        condition = TestEvent.field < 10
        assert repr(condition) == "Is(TestEvent.field < 10)"

    def test_less_than_or_equal_condition_repr(self):
        condition = TestEvent.field <= 10
        assert repr(condition) == "Is(TestEvent.field <= 10)"

    def test_greater_than_condition_repr(self):
        condition = TestEvent.field > 10
        assert repr(condition) == "Is(TestEvent.field > 10)"

    def test_greater_than_or_equal_condition_repr(self):
        condition = TestEvent.field >= 10
        assert repr(condition) == "Is(TestEvent.field >= 10)"

    def test_is_condition_repr(self):
        condition = TestEvent.field.is_(None)
        assert repr(condition) == "Is(TestEvent.field is None)"


class TestAndConditionRepr:
    def test_and_condition_repr(self):
        condition = (TestEvent.field > 0) & (TestEvent.field < 10)
        assert repr(condition) == "And(Is(TestEvent.field > 0) & Is(TestEvent.field < 10))"

    def test_and_condition_repr_single(self):
        condition = (TestEvent.field > 0) & (TestEvent.field < 10) & (TestEvent.field != 5)
        expected = "And(Is(TestEvent.field > 0) & Is(TestEvent.field < 10) & Is(TestEvent.field != 5))"
        assert repr(condition) == expected


class TestOrConditionRepr:
    def test_or_condition_repr(self):
        condition = (TestEvent.field < 0) | (TestEvent.field > 10)
        assert repr(condition) == "Or(Is(TestEvent.field < 0) | Is(TestEvent.field > 10))"

    def test_or_condition_repr_single(self):
        condition = (TestEvent.field < 0) | (TestEvent.field > 10) | (TestEvent.field == 5)
        expected = "Or(Is(TestEvent.field < 0) | Is(TestEvent.field > 10) | Is(TestEvent.field == 5))"
        assert repr(condition) == expected


class TestNotConditionRepr:
    def test_not_condition_repr(self):
        condition = ~(TestEvent.field == 5)
        assert repr(condition) == "~Is(TestEvent.field == 5)"


class TextTypeConditionRepr:
    def test_type_condition_repr(self):
        condition = TypeCondition(TestEvent)
        assert repr(condition) == "IsType(TestEvent)"

    def test_type_condition_repr_tuple(self):
        condition = TypeCondition((TestEvent, AnotherEvent))
        assert repr(condition) == "IsType(TestEvent | AnotherEvent)"

    def test_type_condition_repr_union(self):
        condition = TypeCondition(TestEvent | AnotherEvent)
        assert repr(condition) == "IsType(TestEvent | AnotherEvent)"


class TestMixed:
    def test_complex_condition_repr(self):
        condition = ((TestEvent.field >= 0) & (TestEvent.field <= 10)) | (TestEvent.field == 100)
        expected = "Or(And(Is(TestEvent.field >= 0) & Is(TestEvent.field <= 10)) | Is(TestEvent.field == 100))"
        assert repr(condition) == expected

    def test_nested_condition_repr(self):
        condition = ~((TestEvent.field > 0) & (TestEvent.field < 10))
        expected = "~And(Is(TestEvent.field > 0) & Is(TestEvent.field < 10))"
        assert repr(condition) == expected

    def test_class_or_condition_repr(self):
        condition = AnotherEvent | TestEvent
        assert repr(condition) == "Or(IsType(AnotherEvent) | IsType(TestEvent))"

    def test_class_or_with_condition_repr(self):
        condition = AnotherEvent | (TestEvent.field > 10)
        expected = "Or(IsType(AnotherEvent) | Is(TestEvent.field > 10))"
        assert repr(condition) == expected

    def test_condition_or_class_repr(self):
        condition = (TestEvent.field > 10) | AnotherEvent
        expected = "Or(Is(TestEvent.field > 10) | IsType(AnotherEvent))"
        assert repr(condition) == expected
