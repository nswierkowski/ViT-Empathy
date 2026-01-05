from pathlib import Path
from torchvision import transforms
import argparse
from src.etl.etl_processing import ETLProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="ETL pipeline for face emotion dataset"
    )

    parser.add_argument(
        "--image_dir", type=Path, required=True, help="Directory with raw images"
    )

    parser.add_argument(
        "--processed_dir",
        type=Path,
        required=True,
        help="Directory to store processed splits",
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for DataLoaders"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for deterministic splitting",
    )

    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    transform = transforms.Compose([transforms.ToTensor()])

    etl = ETLProcessor(
        image_dir=args.image_dir,
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        random_state=args.random_state,
        transform=transform,
    )

    train_loader, val_loader, test_loader = etl.run()
