#!/bin/bash
set -e

cd "$(dirname "$0")"

# Split dataset1
# siin-trainer split-dataset --dataset ./dataset1 --val 0.25 --test 0.25 --seed 1

# # Merge dataset1 & dataset2
# siin-trainer merge-datasets --datasets ./dataset1  --datasets  ./dataset2 --output ./merged_dataset

# # Visualize dataset1
# siin-trainer visualize-dataset --dataset ./dataset1 --output ./dataset1_visualization

# # Filter dataset1
# siin-trainer filter-objects --dataset ./dataset1 --objects "0" --objects "8" --output ./dataset1_filtered
# # # Visualize filtered dataset1
# siin-trainer visualize-dataset --dataset ./dataset1_filtered --output ./dataset1_filtered_visualization

# # Convert dataset1 to COCO format
# siin-trainer yolo-to-coco --dataset ./dataset1 --output-json ./coco-dataset/here.json

# # Convert dataset1 to YOLO format
# siin-trainer coco-to-yolo --coco-json ./coco-dataset/here.json --output-dir ./yolo-dataset
# # # Visualize YOLO dataset
# siin-trainer visualize-dataset --dataset ./yolo-dataset --output ./yolo_dataset_visualization

# # Train model on dataset1
# siin-trainer train-ultralytics --data ./dataset1/data.yaml --model yolov8n.pt --device "cpu" --epochs 1 --batch 2

# # Download a dataset
# siin-trainer download-dataset --url "https://datasets.siin.ai/Barcode/latest/barcode-recognition.zip" --dir ./barcode_recognition_dataset

# Extract frames from video
# siin-trainer extract-frames --video ./test-video.mp4 --output-dir ./extracted-frames --similarity-threshold 0.95
