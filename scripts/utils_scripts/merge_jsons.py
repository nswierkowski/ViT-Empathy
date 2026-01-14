import argparse
import json
from pathlib import Path


def merge_json_files(input_dir: Path, output_file: Path):
    merged_data = {}

    for file_path in sorted(input_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in file: {file_path}") from e

                if not isinstance(data, dict):
                    raise TypeError(
                        f"Expected JSON object (dict) in {file_path}, got {type(data).__name__}"
                    )

                merged_data.update(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Merge all JSON files in a folder into a single JSON dictionary"
    )
    parser.add_argument(
        "--input_folder",
        type=Path,
        help="Path to the folder containing JSON files",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        help="Path to the output merged JSON file",
    )

    args = parser.parse_args()

    if not args.input_folder.exists() or not args.input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {args.input_folder}")

    merge_json_files(args.input_folder, args.output_json)


if __name__ == "__main__":
    main()
