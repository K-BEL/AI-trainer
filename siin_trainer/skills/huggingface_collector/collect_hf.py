import sys
import argparse
import subprocess
from pathlib import Path

# --- Configuration ---
DEFAULT_ROOT = Path("tests/dataset_hf")

import zipfile

def check_auto_train(root_path):
    print(f"🚀 Download complete for {root_path}. Processing data...")
    
    # Check for zip files (common in HF datasets)
    data_dir = root_path / "data"
    if data_dir.exists():
        for zip_path in data_dir.glob("*.zip"):
            print(f"📦 Unzipping {zip_path.name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root_path)

    # Check for COCO annotations (Roboflow exports)
    coco_json = root_path / "_annotations.coco.json"
    if coco_json.exists() and not (root_path / "data.yaml").exists():
        print(f"📄 Found COCO annotations. Converting to YOLO...")
        from siin_trainer.datasets.coco_to_yolo import convert_coco_json_to_yolo, generate_simple_data_yaml
        convert_coco_json_to_yolo(coco_json, root_path, root_path)
        generate_simple_data_yaml(root_path, ["license_plate"])

    # Attempt to find data.yaml in the downloaded folder
    data_yaml = root_path / "data.yaml"
    if not data_yaml.exists():
        # Look for it one level deeper
        found = list(root_path.rglob("data.yaml"))
        if found:
            data_yaml = found[0]
            print(f"📂 Found data.yaml at {data_yaml}")
        else:
            print(f"⚠️ Warning: No data.yaml found in {root_path}. Manual configuration required.")
            return

    subprocess.run([
        sys.executable, "-m", "siin_trainer.cli", "train-ultralytics",
        "--data", str(data_yaml),
        "--epochs", "50",
        "--device", "auto"
    ])

def main():
    parser = argparse.ArgumentParser(description="Hugging Face Dataset Collector Skill")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID")
    parser.add_argument("--subset", type=str, default=None, help="Subset pattern")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="Local root directory")

    args = parser.parse_args()
    root_path = Path(args.root)

    # 1. Trigger the download-hf CLI command
    cmd = [
        sys.executable, "-m", "siin_trainer.cli", "download-hf",
        "--repo-id", args.repo_id,
        "--root", args.root
    ]
    if args.subset:
        cmd.extend(["--subset", args.subset])

    subprocess.run(cmd, check=True)

    # 2. Trigger auto-training
    check_auto_train(root_path)

if __name__ == "__main__":
    main()
