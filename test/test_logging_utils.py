import io
import logging

from autogen.logging_utils import EventStreamHandler, event_print


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_event_print_with_custom_logger_and_handler() -> None:
    handler = _RecordingHandler()
    logger = logging.getLogger("ag2.event.processor.test")
    logger.handlers = [handler]
    logger.propagate = False

    event_print("hello", "world", sep="-", end="!", flush=False, logger=logger, level=logging.WARNING)

    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.getMessage() == "hello-world"
    assert getattr(record, "ag2_event_end") == "!"
    assert getattr(record, "ag2_event_flush") is False
    assert record.levelno == logging.WARNING


def test_event_print_default_logger_respects_end_and_flush() -> None:
    stream = io.StringIO()
    logger = logging.getLogger("ag2.event.processor")
    old_handlers = logger.handlers[:]
    old_propagate = logger.propagate
    try:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        handler = EventStreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        event_print("structured", "output", sep="|", end="END", flush=True)

        assert stream.getvalue() == "structured|outputEND"
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.propagate = old_propagate
