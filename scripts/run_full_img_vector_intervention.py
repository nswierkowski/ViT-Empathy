import argparse
from pathlib import Path
import pandas as pd

from torchvision import transforms

from src.etl.etl_processing import ETLProcessor
from src.analysis.emotion_vector_retriever.full_img_intervention import (
    FullImageVectorInterventionExp,
)

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
        description="Test causal effect of FULL-IMAGE emotion vectors using frozen linear probes"
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
        help="Path to full-image emotion_pair_stats.pt",
    )

    p.add_argument(
        "--emotion_from",
        type=str,
        required=True,
        choices=list(EMOTION_TO_IDX.keys()),
    )

    p.add_argument(
        "--emotion_to",
        type=str,
        required=True,
        choices=list(EMOTION_TO_IDX.keys()),
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling factor for full-image vector intervention",
    )

    p.add_argument(
        "--normalize_vector",
        action="store_true",
        help="L2-normalize emotion vectors before applying",
    )

    p.add_argument(
        "--exclude_cls",
        action="store_true",
        help="Apply vector only to patch tokens (exclude CLS)",
    )

    p.add_argument(
        "--path_to_save_results",
        type=Path,
        required=True,
        help="Directory to store intervention results",
    )

    p.add_argument(
        "--random_state",
        type=int,
        default=42,
    )

    p.add_argument(
        "--intervene_layer",
        type=int,
        default=None,
        help="Layer at which full-image vector is injected",
    )

    p.add_argument(
        "--sweep_intervene_layers",
        action="store_true",
        help="Sweep intervention layer from start_layer to last_layer",
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

    if args.split == "train":
        dataloader = train_loader
    elif args.split == "val":
        dataloader = val_loader
    elif args.split == "test":
        dataloader = test_loader
    else:
        raise ValueError(f"Unknown split: {args.split}")

    intervene_layers = []

    if args.sweep_intervene_layers:
        intervene_layers = list(range(args.start_layer, args.last_layer + 1))
    else:
        if args.intervene_layer is None:
            raise ValueError(
                "--intervene_layer must be specified unless --sweep_intervene_layers is used"
            )
        intervene_layers = [args.intervene_layer]

    all_dfs = []

    for intervene_layer in intervene_layers:
        print("=" * 60)
        print(f"Running FULL-IMAGE intervention at layer {intervene_layer}")
        print("=" * 60)

        exp = FullImageVectorInterventionExp(
            model_name=args.model_name,
            start_layer=args.start_layer,
            last_layer=args.last_layer,
            intervene_layer=intervene_layer,
            dataloader=dataloader,
            preds_path=args.preds_path,
            path_to_save_results=args.path_to_save_results,
            vectors_path=args.vectors_path,
            emotion_from=EMOTION_TO_IDX[args.emotion_from],
            emotion_to=EMOTION_TO_IDX[args.emotion_to],
            alpha=args.alpha,
            normalize_vector=args.normalize_vector,
            exclude_cls=args.exclude_cls,
            device=args.device,
        )

        df = exp.run()
        df["sweep_intervene_layer"] = intervene_layer
        all_dfs.append(df)

    df = pd.concat(all_dfs, ignore_index=True)

    out_path = (
        args.path_to_save_results
        / f"fullimg_intervention_from_{args.emotion_from}_to_{args.emotion_to}.csv"
    )
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print("FULL-IMAGE VECTOR INTERVENTION FINISHED")
    print("=" * 60)
    print(f"Emotion: {args.emotion_from} → {args.emotion_to}")
    print(f"Alpha: {args.alpha}")
    print(f"Samples evaluated: {len(df)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
