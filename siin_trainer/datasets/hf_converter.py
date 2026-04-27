import os
from pathlib import Path
from tqdm import tqdm
try:
    from datasets import load_dataset
    from PIL import Image
except ImportError:
    pass

def convert_hf_to_yolo(repo_id, output_root, limit=500):
    """
    Downloads a Hugging Face dataset and converts it to YOLO format.
    Ensures compatibility with siin-trainer training pipeline.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    img_dir = output_root / "images"
    lbl_dir = output_root / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)

    print(f"🔄 Converting Hugging Face dataset '{repo_id}' to YOLO format...")
    
    # Load dataset using HF datasets library
    dataset = load_dataset(repo_id, name="full")
    
    # Map splits
    splits = {
        'train': dataset['train'],
        'val': dataset['validation'] if 'validation' in dataset else dataset['valid'],
        'test': dataset['test']
    }

    class_names = []
    
    for split_name, ds in splits.items():
        print(f"📦 Processing split: {split_name} ({len(ds)} images)")
        
        # YOLO structure often prefers all items in images/ and labels/ 
        # But for large datasets we should preserve splits
        split_img_dir = img_dir / split_name
        split_lbl_dir = lbl_dir / split_name
        split_img_dir.mkdir(exist_ok=True)
        split_lbl_dir.mkdir(exist_ok=True)

        count = 0
        for i, item in enumerate(tqdm(ds)):
            if limit and count >= limit:
                break
            
            image = item['image']
            image_id = f"{split_name}_{i}"
            image_path = split_img_dir / f"{image_id}.jpg"
            label_path = split_lbl_dir / f"{image_id}.txt"
            
            # Save image
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path)
            
            w, h = image.size
            
            # Save YOLO labels
            with open(label_path, "w") as f:
                if 'objects' in item:
                    objs = item['objects']
                    for j in range(len(objs['bbox'])):
                        # COCO uses [x, y, width, height] in pixels
                        # YOLO uses [class, x_center, y_center, width, height] normalized
                        bbox = objs['bbox'][j]
                        cat_id = objs['category'][j]
                        
                        # Note: Some HF datasets use different formats for bbox.
                        # keremberke/license-plate-object-detection uses [x, y, w, h]
                        x, y, bw, bh = bbox
                        
                        x_center = (x + bw/2) / w
                        y_center = (y + bh/2) / h
                        nw = bw / w
                        nh = bh / h
                        
                        f.write(f"{cat_id} {x_center} {y_center} {nw} {nh}\n")
            count += 1
                        
            # Collect class names (assuming they are consistent across items)
            if not class_names and 'objects' in item:
                # This depends on dataset features, often found in dataset.features
                pass

    # Create data.yaml
    create_data_yaml(output_root, repo_id)

def create_data_yaml(root, repo_id):
    import yaml
    
    # Get classes from the first available image if possible, or use default
    classes = ["license_plate"] # Default for license plate detection
    
    data = {
        'path': str(root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    with open(root / "data.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml at {root / 'data.yaml'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    convert_hf_to_yolo(args.repo_id, args.root)
