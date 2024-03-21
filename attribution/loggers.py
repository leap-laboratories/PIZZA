from abc import ABC, abstractmethod
from enum import Enum, auto


class Verbosity(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()


class Logger(ABC):
    def __init__(self, min_verbosity_level: Verbosity = Verbosity.INFO):
        self.min_verbosity_level = min_verbosity_level

    def log(self, message: str, verbosity=Verbosity.INFO):
        if verbosity.value >= self.min_verbosity_level.value:
            self._log_message(message, verbosity)

    @abstractmethod
    def _log_message(self, message: str, verbosity: Verbosity):
        raise """Log to your preferred logger."""


class ConsoleLogger(Logger):
    def _log_message(self, message: str, verbosity: Verbosity):
        print(f"{verbosity.name}: {message}")
