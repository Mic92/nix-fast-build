import enum
from dataclasses import dataclass, field


class ResultType(enum.Enum):
    EVAL = enum.auto()
    BUILD = enum.auto()
    UPLOAD = enum.auto()
    DOWNLOAD = enum.auto()
    CACHIX = enum.auto()
    ATTIC = enum.auto()
    NIKS3 = enum.auto()


@dataclass
class Result:
    result_type: ResultType
    attr: str
    success: bool
    duration: float
    error: str | None
    log_output: str | None = None
    outputs: dict[str, str] | None = None

    def as_dict(self) -> dict:
        return {
            "type": self.result_type.name,
            "attr": self.attr,
            "success": self.success,
            "duration": self.duration,
            "error": self.error,
            **({"outputs": self.outputs} if self.outputs is not None else {}),
        }


@dataclass
class Summary:
    successes: int = 0
    failures: int = 0
    failed_attrs: list[str] = field(default_factory=list)
