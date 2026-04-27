import json
import os
from pathlib import Path
from tqdm import tqdm

def convert_coco_json_to_yolo(json_path, output_dir, img_dir, limit=0):
    """
    Converts a COCO JSON annotation file to YOLO .txt files.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Map category IDs
    # keremberke uses 1 for license_plate. YOLO wants 0
    cat_id_map = {cat['id']: i for i, cat in enumerate(data['categories'])}
    
    # Map image IDs to filenames and sizes
    images = {img['id']: img for img in data['images']}
    
    # Process annotations
    print(f"🔄 Converting COCO annotations from {json_path}...")
    
    processed_images = set()
    for ann in tqdm(data['annotations']):
        img_id = ann['image_id']
        if img_id not in images:
            continue
            
        if limit > 0 and len(processed_images) >= limit and img_id not in processed_images:
            continue
            
        processed_images.add(img_id)
        img_info = images[img_id]
        filename = img_info['file_name']
        w = img_info['width']
        h = img_info['height']
        
        # YOLO path
        label_path = Path(output_dir) / f"{Path(filename).stem}.txt"
        
        # COCO bbox: [x, y, width, height] in pixels
        bbox = ann['bbox']
        cat_id = ann['category_id']
        yolo_cat = cat_id_map.get(cat_id, 0)
        
        x, y, bw, bh = bbox
        x_center = (x + bw/2) / w
        y_center = (y + bh/2) / h
        nw = bw / w
        nh = bh / h
        
        with open(label_path, 'a') as f:
            f.write(f"{yolo_cat} {x_center} {y_center} {nw} {nh}\n")

def generate_simple_data_yaml(root, classes):
    import yaml
    data = {
        'path': str(Path(root).absolute()),
        'train': '.', # Images are in the root for this specific HF unzip
        'val': '.',
        'names': {i: name for i, name in enumerate(classes)}
    }
    with open(Path(root) / "data.yaml", "w") as f:
        yaml.dump(data, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    convert_coco_json_to_yolo(args.json, args.root, args.root)
    generate_simple_data_yaml(args.root, ["license_plate"])
