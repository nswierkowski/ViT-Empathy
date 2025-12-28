import argparse
from pathlib import Path
from src.analysis.cka_compare import CKAComparer
from src.utils.cka.filter_strategies import EmotionFilterStrategy, NoFilterStrategy
import pandas as pd
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Run CKA similarity on two cached ViT activation files")
    parser.add_argument("--pt1", type=Path, required=True, help="Path to first .pt file")
    parser.add_argument("--pt2", type=Path, required=True, help="Path to second .pt file")
    parser.add_argument("--output_dir", type=Path, default=Path("data/analysis"), help="CKA results save dir")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--filter_emotions", type=int, nargs="*", default=None, help="Optional emotion class indices to filter")
    parser.add_argument("--overwrite", action="store_true", help="Recompute if CSV exists")
    return parser.parse_args()

def main():
    args = parse_args()

    exp_name = f"{args.pt1.stem}_VS_{args.pt2.stem}"
    save_path = args.output_dir / f"{exp_name}_cka_results.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not args.overwrite:
        print(f"Found existing CKA results for '{exp_name}', loading...")
        df = pd.read_csv(save_path)
        print(df.head())
        return

    print(f"Running CKA comparison: {args.pt1.name}  VS  {args.pt2.name}")

    act1 = torch.load(args.pt1, map_location=args.device)
    act2 = torch.load(args.pt2, map_location=args.device)

    comparer = CKAComparer(
        features1=act1["features"],
        features2=act2["features"],
        labels=act1["labels"].tolist(),
    )

    if args.filter_emotions is not None:
        comparer.set_filter_strategy(EmotionFilterStrategy(args.filter_emotions))
    else:
        comparer.set_filter_strategy(NoFilterStrategy())

    df = comparer.compare_layers()
    df.to_csv(save_path, index=False)

    print(f"Saved CKA results to: {save_path}")
    print(df.head())


if __name__ == "__main__":
    main()
