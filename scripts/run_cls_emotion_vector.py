import argparse
from pathlib import Path

from torchvision import transforms

from src.etl.etl_processing import ETLProcessor
from src.analysis.emotion_vector_retriever.count_cls_emotion_vector import CLSEmotionMeanVector

EMOTION_TO_IDX = {
    "neutrality": 0,
    "happiness": 1,
    "sadness": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5
}

def build_parser():
    p = argparse.ArgumentParser(
        description="Compute mean/std CLS emotion vectors from cached features"
    )

    p.add_argument(
        "--features_dir",
        type=Path,
        required=True,
        help="Root directory with cached feature experiments",
    )

    p.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Feature extraction experiment name (subdir in features_dir)",
    )

    p.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Which split to use",
    )

    p.add_argument(
        "--start_layer",
        type=int,
        required=True,
        help="First ViT layer index",
    )

    p.add_argument(
        "--last_layer",
        type=int,
        required=True,
        help="Last ViT layer index (inclusive)",
    )

    p.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize mean emotion vectors",
    )

    p.add_argument(
        "--path_to_save_results",
        type=Path,
        required=True,
        help="Directory to store emotion_pair_stats.pt",
    )

    return p


def main():
    args = build_parser().parse_args()
    args.path_to_save_results.mkdir(parents=True, exist_ok=True)

    exp = CLSEmotionMeanVector(
        features_path=(args.features_dir / f"{args.split}.pt"),
        start_layer=args.start_layer,
        last_layer=args.last_layer,
        emotion_map=EMOTION_TO_IDX,
        normalize=args.normalize,
        out_path=args.path_to_save_results / "emotion_pair_stats.pt",
    )

    exp.run()



if __name__ == "__main__":
    main()