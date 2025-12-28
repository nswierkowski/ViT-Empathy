import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class FilterStrategy(ABC):
    @abstractmethod
    def filter(self, labels: List[str]) -> np.ndarray:
        pass
    
class EmotionFilterStrategy(FilterStrategy):
    """Filter activations by one or more emotion labels."""
    def __init__(self, emotions: List[str]):
        self.emotions = set(emotions)

    def filter(self, labels: List[str]) -> np.ndarray:
        return np.array([lab in self.emotions for lab in labels])


class NoFilterStrategy(FilterStrategy):
    """No filtering — keep all samples."""
    def filter(self, labels: List[str]) -> np.ndarray:
        return np.ones(len(labels), dtype=bool)