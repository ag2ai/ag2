from typing import Literal


class SentinelMeta(type):
    """
    A baseclass for sentinels that plays well with type hints.
    Define new sentinels like this:

    ```
    class MY_DEFAULT(metaclass=SentinelMeta):
        pass


    foo: list[str] | None | type[MY_DEFAULT] = MY_DEFAULT
    ```

    Reference: https://stackoverflow.com/questions/69239403/type-hinting-parameters-with-a-sentinel-value-as-the-default
    """

    def __repr__(cls) -> str:
        return f"<{cls.__name__}>"

    def __bool__(cls) -> Literal[False]:
        return False
