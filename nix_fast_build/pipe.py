import os
from types import TracebackType
from typing import Self


class Pipe:
    def __init__(self) -> None:
        fds = os.pipe()
        self.read_file = os.fdopen(fds[0], "rb")
        self.write_file = os.fdopen(fds[1], "wb")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.read_file.close()
        self.write_file.close()
