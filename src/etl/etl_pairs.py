from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PairsDataset(Dataset):
    def __init__(
        self, original_image_path: Path, corrupted_image_path: Path, transform=None
    ):
        self.original_image_path = Path(original_image_path)
        self.corrupted_image_path = Path(corrupted_image_path)
        self.transform = transform

    def __len__(self):
        return 1

    def _parse_filename(self, path: Path):
        person_id, age, sex, emotion, version = path.stem.split("_")
        return {
            "path": str(path),
            "person_id": person_id,
            "age": age,
            "sex": sex,
            "emotion": emotion,
            "version": version,
        }

    def _load_image(self, path: Path):
        image = Image.open(path).convert("RGB")
        return self.transform(image) if self.transform else image

    def __getitem__(self, idx):
        original_image = self._load_image(self.original_image_path)
        corrupted_image = self._load_image(self.corrupted_image_path)
        return {
            "original_image": original_image,
            "corrupted_image": corrupted_image,
            "original_metadata": self._parse_filename(self.original_image_path),
            "corrupted_metadata": self._parse_filename(self.corrupted_image_path),
        }


def create_pairs_dataloader(
    original_image_path: Path,
    corrupted_image_path: Path,
    transform=None,
    batch_size: int = 1,
    num_workers: int = 0,
):
    dataset = PairsDataset(
        original_image_path=original_image_path,
        corrupted_image_path=corrupted_image_path,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
