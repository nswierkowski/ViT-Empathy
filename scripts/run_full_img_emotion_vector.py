import argparse
from pathlib import Path

from torchvision import transforms

from src.etl.etl_processing import ETLProcessor
from src.analysis.emotion_vector_retriever.count_full_img_emotion_vector import FullImageEmotionMeanVector

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
    
    p.add_argument(
        "--device",
        type=Path,
        required=False,
        default='cuda',
        help="Device (cpu or cuda)",
    )
    
    p.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="Processed dataset directory (ETL output)",
    )

    p.add_argument(
        "--image_dir",
        type=Path,
        required=True,
        help="Root directory with raw images",
    )
    
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--random_state",
        type=Path,
        required=False,
        default=42,
        help="Random State value",
    )
    
    p.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
    )

    return p


def main():
    args = build_parser().parse_args()
    args.path_to_save_results.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    
    etl = ETLProcessor(
        image_dir=args.image_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        transform=transform,
    )
    
    train, _, _ = etl.run()

    exp = FullImageEmotionMeanVector(
        model_name=args.model_name,
        dataloader=train,
        start_layer=args.start_layer,
        last_layer=args.last_layer,
        emotion_map=EMOTION_TO_IDX,
        normalize=args.normalize,
        out_path=args.path_to_save_results / "emotion_pair_stats.pt",
        device=args.device
    )
    
    exp.run()



if __name__ == "__main__":
    main()