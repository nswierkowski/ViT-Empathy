import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from src.utils.cka.linear_cka_method import linear_cka_torch

class CompareStrategy(ABC):
    @abstractmethod
    def compare(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        pass


class ActivationCompareStrategy(CompareStrategy):
    """CKA on raw activation matrices."""
    def compare(self, X, Y) -> float:
        return linear_cka_torch(X, Y)


class MeanActivationCompareStrategy(CompareStrategy):
    """CKA between mean activation vectors of two groups."""
    def compare(self, X, Y) -> float:
        return linear_cka_torch(X, Y)


class CrossSexCompareStrategy(CompareStrategy):
    """Compare mean activations of one emotion between male and female."""
    def __init__(self, emotion: str):
        self.emotion = emotion

    def compare(self, features1, features2, metadata) -> pd.DataFrame:
        mask_m = np.array([(e == self.emotion and s == "m") for e, s in zip(metadata["emotion"], metadata["sex"])])
        mask_f = np.array([(e == self.emotion and s == "f") for e, s in zip(metadata["emotion"], metadata["sex"])])
        
        rows = []
        for layer in features1:
            Xm = features1[layer][mask_m]
            Yf = features2[layer][mask_f]
            if Xm.shape[0] < 2 or Yf.shape[0] < 2:
                cka = 0.0
            else:
                n = min(Xm.shape[0], Yf.shape[0])
                idx_m = np.random.choice(Xm.shape[0], n, replace=False)
                idx_f = np.random.choice(Yf.shape[0], n, replace=False)

                Xm_eq = Xm[idx_m]
                Yf_eq = Yf[idx_f]
                cka = linear_cka_torch(Xm_eq, Yf_eq)
            rows.append({"layer": layer, "cka_mean_male_vs_female": cka})
        return pd.DataFrame(rows)


class CrossAgeCompareStrategy(CompareStrategy):
    """Compare mean activations of one emotion between two age categories."""
    def __init__(self, emotion: str, bin1: str, bin2: str):
        self.emotion = emotion
        self.bin1 = bin1
        self.bin2 = bin2

    def compare(self, features1, features2, metadata) -> pd.DataFrame:
        mask1 = np.array([(e == self.emotion and a == self.bin1) for e, a in zip(metadata["emotion"], metadata["age"])])
        mask2 = np.array([(e == self.emotion and a == self.bin2) for e, a in zip(metadata["emotion"], metadata["age"])])

        rows = []
        for layer in features1:
            X1 = features1[layer][mask1]
            X2 = features2[layer][mask2]
            if X1.shape[0] < 2 or X2.shape[0] < 2:
                cka = 0.0
            else:
                cka = linear_cka_torch(X1, X2)
            rows.append({"layer": layer, f"cka_mean_{self.bin1}_vs_{self.bin2}": cka})
        return pd.DataFrame(rows)


class CrossEmotionMeanCompareStrategy(CompareStrategy):
    """Compare mean activations of emotion A vs mean activations of emotion B (same model or different models)."""
    def __init__(self, emotion1: str, emotion2: str):
        self.emotion1 = emotion1
        self.emotion2 = emotion2

    def compare(self, features1, features2, metadata) -> pd.DataFrame:
        mask1 = np.array([e == self.emotion1 for e in metadata["emotion"]])
        mask2 = np.array([e == self.emotion2 for e in metadata["emotion"]])

        rows = []
        for layer in features1:
            X = features1[layer][mask1]
            Y = features2[layer][mask2]
            if X.shape[0] < 2 or Y.shape[0] < 2:
                cka = 0.0
            else:
                cka = linear_cka_torch(X, Y)
            rows.append({"layer": layer, "cka_mean_emotion_compare": cka, "emotion1": self.emotion1, "emotion2": self.emotion2})
        return pd.DataFrame(rows)
