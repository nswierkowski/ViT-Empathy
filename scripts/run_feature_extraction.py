import argparse
from pathlib import Path

import torch
from torchvision import transforms
from transformers import ViTModel

from src.etl.etl_processing import ETLProcessor
from src.analysis.feature_extraction import FeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature extraction for representation analysis"
    )

    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)

    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Human-readable experiment name (used for caching)",
    )

    parser.add_argument(
        "--cache_root",
        type=Path,
        default=Path("analysis/features"),
        help="Root directory for cached features",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="HuggingFace model name",
    )

    parser.add_argument(
        "--pooling",
        type=str,
        choices=["cls", "mean"],
        default="cls",
        help="Pooling strategy for ViT tokens",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached features if they exist",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    etl = ETLProcessor(
        image_dir=args.image_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
    )

    train_loader, val_loader, test_loader = etl.run()

    model = ViTModel.from_pretrained(
        args.model_name,
        output_hidden_states=True
    ).to(device)


    for split_name, loader in [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ]:
        extractor = FeatureExtractor(
            model=model,
            dataloader=loader,
            split_name=split_name,
            experiment_name=args.experiment_name,
            cache_root=args.cache_root,
            pooling=args.pooling,
            device=device,
        )

        extractor.extract(overwrite=args.overwrite)


if __name__ == "__main__":
    main()
