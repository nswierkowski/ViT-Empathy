from dataclasses import dataclass
from pathlib import Path


@dataclass
class ImageSample:
    path: Path
    person_id: str
    age: str
    sex: str
    emotion: str
    version: str
