import torch
import pandas as pd
from typing import List, Dict
from src.utils.cka.linear_cka_method import linear_cka_torch
from src.utils.cka.filter_strategies import FilterStrategy, NoFilterStrategy
from src.utils.cka.compare_strategies import CompareStrategy, ActivationCompareStrategy
from tqdm import tqdm

class CKAComparer:
    def __init__(
        self,
        features1: dict,
        features2: dict,
        labels: List[int],
        compare_strategy: CompareStrategy = ActivationCompareStrategy(),
        filter_strategy: FilterStrategy = NoFilterStrategy(),
    ):
        self.f1 = features1
        self.f2 = features2
        self.labels = labels
        self.compare_strategy = compare_strategy
        self.filter_strategy = filter_strategy

    def compare_layers(self) -> pd.DataFrame:
        mask = self.filter_strategy.filter(self.labels)
        common_layers = sorted(set(self.f1.keys()) & set(self.f2.keys()))
        results = []
        for layer in tqdm(common_layers, desc="Computing CKA per layer"):
            X = self.f1[layer][mask]
            Y = self.f2[layer][mask]
            if X.shape[0] < 2:
                cka_val = 0.0
            else:
                cka_val = self.compare_strategy.compare(X, Y)
            results.append({"layer": layer, "cka": cka_val, "num_samples": int(mask.sum())})
        return pd.DataFrame(results)

    def set_filter_strategy(self, strategy: FilterStrategy):
        self.filter_strategy = strategy

    def set_compare_strategy(self, strategy: CompareStrategy):
        self.compare_strategy = strategy