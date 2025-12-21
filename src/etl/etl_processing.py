from pathlib import Path
from collections import defaultdict
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataset.emotion_dataset import EmotionDataset
from src.dataset.datamodel import ImageSample
from pathlib import Path


class ETLProcessor:
    def __init__(
        self,
        image_dir: Path,
        processed_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        random_state: int = 42,
        transform=None,
    ):
        self.image_dir = image_dir
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.transform = transform

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.train_path = self.processed_dir / "train.pt"
        self.val_path = self.processed_dir / "val.pt"
        self.test_path = self.processed_dir / "test.pt"

    def parse_filename(self, path):
        person_id, age, sex, emotion, version = path.stem.split("_")
        return ImageSample(
            path=path,
            person_id=person_id,
            age=age,
            sex=sex,
            emotion=emotion,
            version=version,
        )

    def load_samples(self):
        samples = [self.parse_filename(p) for p in self.image_dir.glob("*.jpg")]

        persons = defaultdict(list)
        for s in samples:
            persons[s.person_id].append(s)

        return samples, persons

    def split_by_person(self, persons, train_ratio=0.7):
        rows = []
        for pid, imgs in persons.items():
            rows.append({"person_id": pid, "age": imgs[0].age, "sex": imgs[0].sex})

        df = pd.DataFrame(rows)
        df["stratify"] = df["age"] + "_" + df["sex"]

        train_df, temp_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            stratify=df["stratify"],
            random_state=self.random_state,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df["stratify"],
            random_state=self.random_state,
        )

        return train_df, val_df, test_df

    def collect_samples(self, split_df, persons):
        out = []
        for pid in split_df["person_id"]:
            out.extend(persons[pid])
        return out

    def _processed_exists(self):
        return (
            self.train_path.exists()
            and self.val_path.exists()
            and self.test_path.exists()
        )

    def _save(self, train, val, test):
        torch.save(train, self.train_path)
        torch.save(val, self.val_path)
        torch.save(test, self.test_path)

    def _load(self):
        return (
            torch.load(self.train_path),
            torch.load(self.val_path),
            torch.load(self.test_path),
        )

    def _build_dataloaders(self, train_samples, val_samples, test_samples):
        train_ds = EmotionDataset(train_samples, self.transform)
        val_ds = EmotionDataset(val_samples, self.transform)
        test_ds = EmotionDataset(test_samples, self.transform)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader, val_loader, test_loader

    def run(self):
        if self._processed_exists():
            train_samples, val_samples, test_samples = self._load()
        else:
            _, persons = self.load_samples()
            train_df, val_df, test_df = self.split_by_person(persons)

            train_samples = self.collect_samples(train_df, persons)
            val_samples = self.collect_samples(val_df, persons)
            test_samples = self.collect_samples(test_df, persons)

            self._save(train_samples, val_samples, test_samples)

        return self._build_dataloaders(train_samples, val_samples, test_samples)
