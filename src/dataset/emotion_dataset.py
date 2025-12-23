import torch
from torch.utils.data import Dataset
from PIL import Image

class EmotionDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.emotion_to_idx = {
            "n": 0, "h": 1, "s": 2,
            "a": 3, "d": 4, "f": 5
        }
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = Image.open(s["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "emotion": self.emotion_to_idx[s["emotion"]],
            "age": s["age"],
            "sex": s["sex"],
            "person_id": s["person_id"],
            "path": s["path"]
        }
