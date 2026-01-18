import argparse
from pathlib import Path
from src.analysis.linear_probe_analysis import LinearProbeAnalyser
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probing for representation analysis (from cached features)"
    )

    parser.add_argument(
        "--activations_dir",
        type=Path,
        required=True,
        help="Path to experiment activations dir (e.g. data/analysis/vitface_cls_seed42)"
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/analysis"),
        help="Where to save probing results CSV"
    )
    parser.add_argument(
        "--path_preds",
        type=Path,
        default=Path(""),
        help="Where to save predictions CSV"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=6,
        help="Number of emotion classes"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load features & run classifier"
    )

    parser.add_argument(
        "--sklearn_max_iter",
        type=int,
        default=1000,
        help="Max iterations for sklearn LogisticRegression"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, will ignore existing CSV and recompute probing"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    exp_name = args.activations_dir.name
    save_path = args.output_dir / f"{exp_name}_linear_probe_results.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running linear probing for experiment: {exp_name}")

    analyser = LinearProbeAnalyser(
        activations_dir=args.activations_dir,
        num_classes=args.num_classes,
        device=args.device,
        sklearn_max_iter=args.sklearn_max_iter,
        path_preds=args.path_preds
    )

    df = analyser.run()
    df.to_csv(save_path, index=False)

    print(f"Saved linear probing results to: {save_path}")
    print(df.head())

if __name__ == "__main__":
    main()
