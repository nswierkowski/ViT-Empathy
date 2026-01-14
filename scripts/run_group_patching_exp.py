import argparse
from pathlib import Path

from torchvision import transforms

from src.analysis.patching.group_patch_analysis import GroupComboPatchExp
from src.analysis.patching.group_embeding_patch_analysis import GroupComboEmbeddingPatchExp

from src.etl.etl_pairs import create_pairs_dataloader


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run single patching experiment analysis"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="Model name identifier",
    )
    parser.add_argument("--start_layer", type=int, required=True)
    parser.add_argument("--last_layer", type=int, required=True)
    parser.add_argument(
        "--path_to_save_results",
        type=Path,
        required=True,
        help="Directory for outputs and metadata",
    )
    parser.add_argument(
        "--original_image_path",
        type=Path,
        required=True,
        help="Path to the original image file",
    )
    parser.add_argument(
        "--corrupted_image_path",
        type=Path,
        required=True,
        help="Path to the corrupted image file",
    )
    parser.add_argument(
        "--preds_path",
        type=Path,
        default=None,
        help="Optional path to predictions directory",
    )
    parser.add_argument(
        "--json_path",
        type=Path,
        default=None,
        help="path to json file",
    )
    parser.add_argument("--embedding", action="store_true", help="Enable embedding patching")
    parser.add_argument("--use_cls_token", action="store_true", help="Enable CLS token patching")

    parser.add_argument("--device", type=str, default="cuda")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataloader = create_pairs_dataloader(
        original_image_path=args.original_image_path,
        corrupted_image_path=args.corrupted_image_path,
        transform=transform,
    )
    image_name = args.original_image_path.stem + args.original_image_path.suffix
    print("image name:", image_name)

    print(args)
    if args.embedding:
        exp = GroupComboEmbeddingPatchExp(
            model_name=args.model_name,
            start_layer=args.start_layer,
            last_layer=args.last_layer,
            path_to_save_results=args.path_to_save_results,
            dataloader=dataloader,
            preds_path=args.preds_path,
            device=args.device,
            groups_json_path=args.json_path,
            file_key_in_batch=image_name,
            use_cls_token=args.use_cls_token
        )
    else:
        print("Running group patching experiment...")
        exp = GroupComboPatchExp(
            model_name=args.model_name,
            start_layer=args.start_layer,
            last_layer=args.last_layer,
            path_to_save_results=args.path_to_save_results,
            dataloader=dataloader,
            preds_path=args.preds_path,
            device=args.device,
            groups_json_path=args.json_path,
            file_key_in_batch=image_name,
            use_cls_token=args.use_cls_token)
    exp.run()


if __name__ == "__main__":
    main()
