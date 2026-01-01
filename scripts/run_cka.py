import argparse
from pathlib import Path
import pandas as pd
import torch
from src.analysis.cka_compare import CKAComparer
from src.utils.cka.filter_strategies import (
    NoFilterStrategy, EmotionNameFilterStrategy, SexFilterStrategy, AgeCategoryFilterStrategy
)
from src.utils.cka.compare_strategies import (
    ActivationCompareStrategy, CrossSexCompareStrategy, CrossAgeCompareStrategy, CrossEmotionMeanCompareStrategy
)

EMOTION_TO_IDX = {
    "neutrality": 0,
    "happiness": 1,
    "sadness": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5
}

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--pt1", type=Path, required=True)
    p.add_argument("--pt2", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, default=Path("data/analysis/comparing"))
    p.add_argument("--experiment_name", type=str, default="default")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--compare_emotion_means", type=str, nargs=2, default=None,
                   help="Emotion names A vs B (mean CKA)")
    p.add_argument("--compare_sex_means", type=str, default=None,
                   help="Emotion name to compare male vs female means")
    p.add_argument("--compare_age_means", type=str, nargs=3, default=None,
                   help="Emotion + age_bin1 + age_bin2")

    p.add_argument("--filter_emotion", type=str, default=None)
    p.add_argument("--filter_sex", type=str, choices=["m","f"], default=None)
    p.add_argument("--filter_age", type=str, choices=["young","mid","senior"], default=None)

    return p.parse_args()


def main():
    a = parse_args()
    out = a.output_dir / f"{a.experiment_name}_cka.csv"
    a.output_dir.mkdir(parents=True, exist_ok=True)

    if out.exists() and not a.overwrite:
        print(pd.read_csv(out).head())
        return

    act1 = torch.load(a.pt1, map_location=a.device)
    act2 = torch.load(a.pt2, map_location=a.device)

    labels = act1["labels"]
    if isinstance(labels[0], str):
        labels = [EMOTION_TO_IDX[l] for l in labels]

    metadata = {
        "emotion": labels,
        "sex": act1.get("sexes"), 
        "age": act1.get("ages")
    }
    
    print("STARTING CKA COMPARISON")

    if a.compare_emotion_means:
        print("Comparing emotion means:", a.compare_emotion_means)
        e1, e2 = [EMOTION_TO_IDX[name] for name in a.compare_emotion_means]
        print("Emotion indices:", e1, e2)
        df = CrossEmotionMeanCompareStrategy(e1, e2).compare(act1["features"], act2["features"], metadata)

    elif a.compare_sex_means:
        print("Comparing sex means for emotion:", a.compare_sex_means)
        emotion = EMOTION_TO_IDX[a.compare_sex_means]
        df = CrossSexCompareStrategy(emotion).compare(act1["features"], act2["features"], metadata)

    elif a.compare_age_means:
        emo_name, b1, b2 = a.compare_age_means
        emo_idx = EMOTION_TO_IDX[emo_name]
        df = CrossAgeCompareStrategy(emo_idx, b1, b2).compare(act1["features"], act2["features"], metadata)

    else:
        filt = NoFilterStrategy()
        if a.filter_emotion:
            filt = EmotionNameFilterStrategy(a.filter_emotion)
        elif a.filter_sex:
            filt = SexFilterStrategy(a.filter_sex)
        elif a.filter_age:
            filt = AgeCategoryFilterStrategy(a.filter_age)

        cmp = ActivationCompareStrategy()
        comparer = CKAComparer(act1["features"], act2["features"], metadata["emotion"], cmp, filt)
        df = comparer.compare_layers(metadata)

    df.to_csv(out, index=False)
    print(df.head())


if __name__ == "__main__":
    main()
