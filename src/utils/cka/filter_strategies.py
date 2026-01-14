import numpy as np
from abc import ABC, abstractmethod


class FilterStrategy(ABC):
    @abstractmethod
    def filter(self, labels, metadata) -> np.ndarray:
        raise NotImplementedError


class NoFilterStrategy(FilterStrategy):
    def filter(self, labels, metadata=None):
        return np.ones(len(labels), dtype=bool)


class EmotionNameFilterStrategy(FilterStrategy):
    """Keep only samples with a given emotion name."""

    def __init__(self, emotion: str):
        self.emotion = emotion

    def filter(self, labels, metadata):
        emotions = metadata["emotion"]
        print("Filtering for emotion:", self.emotion)
        res = np.array([e == self.emotion for e in emotions])
        print("Number of samples after emotion filter:", int(res.sum()))
        return np.array([e == self.emotion for e in emotions])


class SexFilterStrategy(FilterStrategy):
    """Filter samples by sex column in metadata ('m' or 'f')."""

    def __init__(self, sex: str):
        self.sex = sex

    def filter(self, labels, metadata):
        sexes = metadata["sex"]
        return np.array([s == self.sex for s in sexes])


class AgeCategoryFilterStrategy(FilterStrategy):
    """Filter samples by age category (young/mid/senior)."""

    def __init__(self, category: str):
        self.category = category

    def filter(self, labels, metadata):
        ages = metadata["age"]
        return np.array([a == self.category for a in ages])


class SameEmotionDifferentSexFilterStrategy(FilterStrategy):
    """Filter one emotion, then select only one sex."""

    def __init__(self, emotion: str, sex: str):
        self.emotion = emotion
        self.sex = sex

    def filter(self, labels, metadata):
        emotions = metadata["emotion"]
        sexes = metadata["sex"]
        return np.array(
            [(e == self.emotion and s == self.sex) for e, s in zip(emotions, sexes)]
        )


class SameEmotionAgeBinFilterStrategy(FilterStrategy):
    """Filter one emotion, then select only one age bin."""

    def __init__(self, emotion: str, age_bin: str):
        self.emotion = emotion
        self.age_bin = age_bin

    def filter(self, labels, metadata):
        emotions = metadata["emotion"]
        ages = metadata["age"]
        return np.array(
            [(e == self.emotion and a == self.age_bin) for e, a in zip(emotions, ages)]
        )
