import torch
from pathlib import Path
from tqdm import tqdm
import json


class FeatureExtractor:
    def __init__(
        self,
        model,
        dataloader,
        split_name: str,
        experiment_name: str,
        cache_root: Path,
        pooling: str = "cls",
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.split_name = split_name
        self.experiment_name = experiment_name
        self.cache_root = cache_root
        self.pooling = pooling
        self.device = device

        self.exp_dir = self.cache_root / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _pool(self, hidden_states):
        if self.pooling == "cls":
            return hidden_states[:, 0]
        elif self.pooling == "mean":
            return hidden_states[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _cache_path(self):
        return self.exp_dir / f"{self.split_name}.pt"

    def _metadata_path(self):
        return self.exp_dir / "metadata.json"

    def _save_metadata(self):
        metadata = {
            "experiment_name": self.experiment_name,
            "split": self.split_name,
            "pooling": self.pooling
        }
        with open(self._metadata_path(), "w") as f:
            json.dump(metadata, f, indent=2)

    def _check_metadata(self):
        if not self._metadata_path().exists():
            return True 

        with open(self._metadata_path()) as f:
            meta = json.load(f)

        return meta["pooling"] == self.pooling

    @torch.no_grad()
    def extract(self, overwrite=False):
        cache_path = self._cache_path()

        if cache_path.exists() and not overwrite:
            if not self._check_metadata():
                raise RuntimeError(
                    f"Metadata mismatch for experiment {self.experiment_name}"
                )
            return torch.load(cache_path)

        all_features = {}
        all_labels = []

        for batch in tqdm(self.dataloader, desc=f"Extracting {self.split_name}"):
            images = batch["image"].to(self.device)
            labels = batch["emotion"]

            outputs = self.model(images, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            if not all_features:
                for i in range(len(hidden_states)):
                    all_features[i] = []

            for layer_idx, h in enumerate(hidden_states):
                pooled = self._pool(h)
                all_features[layer_idx].append(pooled.cpu())

            all_labels.append(labels)

        for layer_idx in all_features:
            all_features[layer_idx] = torch.cat(all_features[layer_idx], dim=0)

        labels = torch.cat(all_labels, dim=0)

        data = {
            "features": all_features,
            "labels": labels
        }

        torch.save(data, cache_path)
        self._save_metadata()
        return data
