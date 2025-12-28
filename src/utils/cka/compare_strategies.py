import torch
from abc import ABC, abstractmethod
from src.utils.cka.linear_cka_method import linear_cka_torch


class CompareStrategy(ABC):
    @abstractmethod
    def compare(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        pass


class ActivationCompareStrategy(CompareStrategy):
    """CKA on activation matrices."""
    def compare(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        return linear_cka_torch(X, Y)


class WeightCompareStrategy(CompareStrategy):
    """CKA on single vectors (like classifier weights)."""
    def compare(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        return linear_cka_torch(X[None, :], Y[None, :])