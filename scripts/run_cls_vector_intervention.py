import argparse
from pathlib import Path

from torchvision import transforms

from src.etl.etl_processing import ETLProcessor
from src.analysis.emotion_vector_retriever.cls_intervention import CLSVectorInterventionExp


EMOTION_TO_IDX = {
    "neutrality": 0,
    "happiness": 1,
    "sadness": 2,
    "anger": 3,
    "disgust": 4,
    "fear": 5,
}


def build_parser():
    p = argparse.ArgumentParser(
        description="Test causal effect of CLS emotion vectors using frozen linear probes"
    )

    p.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
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

    p.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to evaluate intervention on",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--start_layer", type=int, required=True)
    p.add_argument("--last_layer", type=int, required=True)

    p.add_argument(
        "--preds_path",
        type=Path,
        required=True,
        help="Path with saved linear probes (lr_layer_cls/)",
    )

    p.add_argument(
        "--vectors_path",
        type=Path,
        required=True,
        help="Path to emotion_pair_stats.pt",
    )
    
    p.add_argument(
        "--emotion_from",
        type=str,
        required=True,
        choices=list(EMOTION_TO_IDX.keys()),
        help="Source emotion",
    )

    p.add_argument(
        "--emotion_to",
        type=str,
        required=True,
        choices=list(EMOTION_TO_IDX.keys()),
        help="Target emotion",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling factor for CLS vector intervention",
    )

    p.add_argument(
        "--normalize_vector",
        action="store_true",
        help="L2-normalize emotion vectors before applying",
    )

    p.add_argument(
        "--path_to_save_results",
        type=Path,
        required=True,
        help="Directory to store intervention results",
    )
    
    p.add_argument(
        "--random_state",
        type=Path,
        required=False,
        default=42,
        help="Random State value",
    )

    p.add_argument("--device", type=str, default="cuda")

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

    train_loader, val_loader, test_loader = etl.run()
    dataloader = None

    if args.split == 'train':
        dataloader = train_loader
    elif args.split == 'val':
        dataloader = val_loader
    elif args.split == 'test':
        dataloader = test_loader
    else:
        raise NotImplementedError(f"Split by {args.split} is not implemented - use train, val or test")

    exp = CLSVectorInterventionExp(
        model_name=args.model_name,
        start_layer=args.start_layer,
        last_layer=args.last_layer,
        dataloader=dataloader,
        preds_path=args.preds_path,
        path_to_save_results=args.path_to_save_results,
        vectors_path=args.vectors_path,
        emotion_from=EMOTION_TO_IDX[args.emotion_from],
        emotion_to=EMOTION_TO_IDX[args.emotion_to],
        alpha=args.alpha,
        normalize_vector=args.normalize_vector,
        device=args.device,
    )

    df = exp.run()

    print("=" * 60)
    print("CLS VECTOR INTERVENTION FINISHED")
    print("=" * 60)
    print(f"Samples evaluated: {len(df)}")
    print(f"Saved to: {args.path_to_save_results / 'cls_vector_intervention.csv'}")


if __name__ == "__main__":
    main()
