import os
import shutil
from pathlib import Path
import random

def reorganize(root_path, val_split=0.2):
    root = Path(root_path)
    
    # Define targets
    train_img = root / "train" / "images"
    train_lbl = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lbl = root / "val" / "labels"
    
    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Find all images
    images = list(root.glob("*.jpg"))
    random.shuffle(images)
    
    split_idx = int(len(images) * (1 - val_split))
    train_set = images[:split_idx]
    val_set = images[split_idx:]
    
    def move_files(files, img_dest, lbl_dest):
        for img_path in files:
            lbl_path = img_path.with_suffix(".txt")
            
            # Move image
            shutil.move(str(img_path), str(img_dest / img_path.name))
            
            # Move label if exists
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(lbl_dest / lbl_path.name))

    print(f"🚚 Moving {len(train_set)} images to train...")
    move_files(train_set, train_img, train_lbl)
    
    print(f"🚚 Moving {len(val_set)} images to val...")
    move_files(val_set, val_img, val_lbl)

    # 3. Update data.yaml
    import yaml
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            
        data['path'] = str(root.absolute())
        data['train'] = "train/images"
        data['val'] = "val/images"
        # delete old test path if it's there
        if 'test' in data:
            data['test'] = "val/images" 

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f)
            
    print("✅ Dataset successfully restructured!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    reorganize(args.root)
