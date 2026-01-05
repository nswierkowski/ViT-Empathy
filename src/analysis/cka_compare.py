import pandas as pd
from tqdm import tqdm


class CKAComparer:
    def __init__(self, features1, features2, labels, compare_strategy, filter_strategy):
        self.f1 = features1
        self.f2 = features2
        self.labels = labels
        self.compare_strategy = compare_strategy
        self.filter_strategy = filter_strategy

    def compare_layers(self, metadata=None) -> pd.DataFrame:
        mask = self.filter_strategy.filter(self.labels, metadata)
        print("Number of samples after filtering:", int(mask.sum()))
        common_layers = sorted(set(self.f1.keys()) & set(self.f2.keys()))
        rows = []

        for layer in tqdm(common_layers, desc="Computing CKA per layer"):
            X = self.f1[layer][mask]
            Y = self.f2[layer][mask]
            if X.shape[0] < 2:
                cka = 0.0
            else:
                cka = self.compare_strategy.compare(X, Y)
            rows.append({"layer": layer, "cka": cka, "num_samples": int(mask.sum())})

        return pd.DataFrame(rows)
