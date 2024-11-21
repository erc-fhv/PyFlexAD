import time
from contextlib import ContextDecorator
from types import TracebackType
from typing import Optional, Type


class Timer(ContextDecorator):

    def __init__(self, name: str = "Timer", n_runs: int = 1) -> None:
        self.__name = name
        self.__n_runs = n_runs

    @staticmethod
    def get_time() -> float:
        return time.perf_counter()

    @classmethod
    def stop_time(cls, start_time: float) -> float:
        return cls.get_time() - start_time

    def __enter__(self):
        self.start_time = self.get_time()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException],
                 traceback: Optional[TracebackType]) -> None:
        self.dt = self.stop_time(self.start_time) / self.__n_runs

    def __repr__(self) -> str:
        return f"{self.__name} took {self.dt} s"
