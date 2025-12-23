import torch
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score


class LinearProbeAnalyser:
    def __init__(
        self,
        activations_dir: Path,
        num_classes: int = 6,
        device: str = "cpu",
        sklearn_max_iter: int = 1000
    ):
        self.activations_dir = activations_dir
        self.num_classes = num_classes
        self.device = device
        self.sklearn_max_iter = sklearn_max_iter

    def _load_split(self, split: str):
        path = self.activations_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Activations for split '{split}' not found")

        return torch.load(path, map_location=self.device)

    def _train_logreg(self, X, y):
        X = X.numpy()
        y = y.numpy()

        clf = LogisticRegression(
            max_iter=self.sklearn_max_iter,
            random_state=self._get_seed(),
            class_weight="balanced",
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1
        )
        clf.fit(X, y)
        return clf

    def _get_seed(self):
        parts = self.activations_dir.name.lower().split("seed")
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        return 42

    def _evaluate_logreg(self, clf, X, y):
        X = X.numpy()
        y = y.numpy()

        preds = clf.predict(X)
        cm = confusion_matrix(y, preds, labels=list(range(self.num_classes)))

        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)

        return acc, bal_acc, cm, len(y)

    def run(self):
        train_data = self._load_split("train")
        val_data   = self._load_split("val")

        results = []

        for layer_idx, X_train in train_data["features"].items():
            y_train = train_data["labels"]

            X_val = val_data["features"][layer_idx]
            y_val = val_data["labels"]

            clf = self._train_logreg(X_train, y_train)

            acc, bal_acc, cm, n = self._evaluate_logreg(clf, X_val, y_val)

            results.append({
                "layer": layer_idx,
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "confusion": cm.flatten().tolist(),
                "confusion_shape": cm.shape,
                "num_samples": n
            })

        return pd.DataFrame(results)
