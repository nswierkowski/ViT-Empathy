import argparse
from pathlib import Path

from torchvision import transforms

from src.etl.etl_processing import ETLProcessor

from src.etl.face_parsing import run as face_parsing_run
from typing import List, Tuple, Union

def parse_id_groups(values: List[str]) -> List[Union[int, Tuple[int, ...]]]:
    groups = []
    for v in values:
        if "," in v:
            group = tuple(int(x) for x in v.split(",") if x.strip() != "")
            if len(group) < 2:
                raise argparse.ArgumentTypeError(
                    f"Group '{v}' must contain at least 2 ids"
                )
            groups.append(group)
        else:
            groups.append(int(v))
    return groups

def parse_args():
    parser = argparse.ArgumentParser(description="Face Parsing -> Patch-to-Class JSON")

    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)

    parser.add_argument(
        "--path_to_save_json",
        type=Path,
        required=True,
        help="Output JSON file path (e.g. analysis/face_patches.json)",
    )

    parser.add_argument(
        "--save_mask",
        action="store_true",
        help="If set, saves predicted masks to a 'masks/' folder next to the JSON",
    )

    parser.add_argument(
        "--ids_to_save",
        nargs="+",
        required=True,
        metavar="ID|ID,ID",
        help=(
            "Face-parser class ids or grouped ids.\n"
            "Examples:\n"
            "  --ids_to_save 4,5 12,13 10\n"
            "  (4,5)=eyes, (12,13)=lips, 10=nose"
        ),
    )


    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=["train", "val", "test"],
        default=["train"],
        help="Which dataset split(s) to process",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    args.ids_to_save = parse_id_groups(args.ids_to_save)
    return args


def main():
    args = parse_args()

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
        transform=transform,
    )

    train_loader, val_loader, test_loader = etl.run()

    split_to_loader = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    out_path: Path = args.path_to_save_json
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        loader = split_to_loader[split]

        if len(args.splits) == 1:
            split_json_path = out_path
        else:
            split_json_path = out_path.with_name(f"{out_path.stem}_{split}{out_path.suffix}")

        face_parsing_run(
            dataloader=loader,
            path_to_save_json=str(split_json_path),
            save_mask=bool(args.save_mask),
            ids_to_save=args.ids_to_save,
        )

        print(f"[OK] Saved {split} results to: {split_json_path}")


if __name__ == "__main__":
    main()
