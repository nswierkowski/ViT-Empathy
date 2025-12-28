import torch
import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score


class LinearProbeAnalyser:
    def __init__(
        self,
        activations_dir: Path,
        num_classes: int = 6,
        device: str = "cpu",
        sklearn_max_iter: int = 1000,
        path_preds: Path | None = None
    ):
        self.activations_dir = activations_dir
        self.num_classes = num_classes
        self.device = device
        self.sklearn_max_iter = sklearn_max_iter
        self.path_preds = path_preds
        self.emotion_map = ["neutrality", "happiness", "sadness", "anger", "disgust", "fear"]

    def _load_split(self, split: str):
        path = self.activations_dir / f"{split}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Activations for split '{split}' not found")
        return torch.load(path, map_location=self.device)

    def _train_logreg(self, X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        clf = LogisticRegression(
            max_iter=self.sklearn_max_iter,
            random_state=self._get_seed(),
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(X, y)
        return clf

    def _get_seed(self):
        parts = self.activations_dir.name.lower().split("seed")
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        return 42

    def _evaluate_logreg(self, clf, X, y):
        X = X.cpu().numpy()
        y = y.cpu().numpy()
        preds = clf.predict(X)
        cm = confusion_matrix(y, preds, labels=list(range(self.num_classes)))

        acc = accuracy_score(y, preds)
        bal_acc = balanced_accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")

        return acc, bal_acc, f1, cm, len(y), preds, y

    def run(self):
        train_data = self._load_split("train")
        val_data = self._load_split("val")

        results = []

        for layer_idx, X_train in train_data["features"].items():
            y_train = train_data["labels"]
            X_val = val_data["features"][layer_idx]
            y_val = val_data["labels"]

            clf = self._train_logreg(X_train, y_train)

            acc, bal_acc, f1, cm, n, val_preds, val_y_true = self._evaluate_logreg(clf, X_val, y_val)

            for class_idx, emotion in enumerate(self.emotion_map):
                results.append({
                    "layer": layer_idx,
                    "emotion": emotion,
                    "class_index": class_idx,
                    "accuracy": acc,
                    "balanced_accuracy": bal_acc,
                    "f1": f1,
                    "confusion": cm.flatten().tolist(),
                    "confusion_shape": cm.shape,
                    "num_samples": n,
                    "weights": clf.coef_[class_idx].tolist(),
                    "bias": clf.intercept_[class_idx]
                })

            if self.path_preds:
                self.path_preds.mkdir(parents=True, exist_ok=True)
                paths = val_data.get("image_paths")
                df_preds = pd.DataFrame({
                    "image_path": paths,
                    "sexes": val_data.get("sexes"),
                    "ages": val_data.get("ages"),
                    "true_label": val_preds,
                    "predicted_label": val_y_true
                })
                df_preds.to_csv(
                    self.path_preds / f"layer_{layer_idx}_predictions.csv",
                    index=False
                )

        return pd.DataFrame(results)
