from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class BaseLLMAttributor(ABC):
    @abstractmethod
    def compute_attributions(
        self, input_text: str, *args, **kwargs
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        pass


class BaseAsyncLLMAttributor(ABC):
    @abstractmethod
    async def compute_attributions(
        self, input_text: str, *args, **kwargs
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        pass


class BaseLogger(ABC):
    @abstractmethod
    def start_experiment(self, *args, **kwargs):
        pass

    @abstractmethod
    def log_attribution(self, *args, **kwargs):
        pass

    @abstractmethod
    def stop_experiment(self):
        pass
