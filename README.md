# AI-trainer

Library to train and evaluate AI vision models with a unified CLI.

Live Demo Dashboard

## Requirements

- Python `3.11` or `3.12`
- Git
- Internet access for dependency/model downloads
- Optional GPU/CUDA setup for faster training/inference
- Optional Weights & Biases account (`wandb`) for experiment tracking
- ONNX export dependencies are excluded from default install

## Setup

### Linux/macOS (uv-based, recommended)

```bash
./scripts/setup-environment.sh
source .venv/bin/activate
siin-trainer --help
```

What the script does:

- Installs `uv` if missing
- Installs Python version from `.python-version`
- Recreates `.venv` in a clean state
- Installs and syncs dependencies
- Installs the package in editable mode

Notes:

- The script intentionally installs `rfdetr[metrics]` (without `onnxexport`) to avoid `onnxsim` CMake build issues on some systems.
- You can activate from any directory with:
  - `source /absolute/path/to/AI-trainer/.venv/bin/activate`

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
siin-trainer --help
```

If `siin-trainer` is not found, run:

```powershell
python -m trainer.cli --help
```

## Update dependencies later

- Linux/macOS (uv):
  - `source .venv/bin/activate`
  - `uv sync`
- Windows (pip):
  - `.\.venv\Scripts\Activate.ps1`
  - `python -m pip install -e .`

## CLI quick start

Show all commands:

```bash
siin-trainer --help
```

Get help for a single command:

```bash
siin-trainer <command> --help
```

## Core commands

### Dataset operations

- Split dataset:

```bash
siin-trainer split-dataset --dataset /path/to/dataset --val 0.15 --test 0.05 --seed 42
```

- Merge datasets:

```bash
siin-trainer merge-datasets --output /path/to/out --datasets /path/ds1 --datasets /path/ds2
```

### Training

- Ultralytics:

```bash
siin-trainer train-ultralytics --data /path/to/data.yaml --model yolov8n --epochs 50 --img-size 640 --batch 16 --device cuda
```

- RF-DETR:

```bash
siin-trainer train-rfdetr --data /path/to/dataset_dir --model RFDETRMedium --epochs 50 --batch-size 16 --device cuda
```

## Backends

- `ultralytics`: YOLO training/evaluation/benchmarking
- `rfdetr`: RF-DETR training/evaluation/benchmarking
- `custom`: alias mode for user checkpoints via `--custom-backend-type`

## Evaluate (tester)

Evaluate a trained checkpoint:

```bash
siin-trainer eval --backend ultralytics --checkpoint /path/to/best.pt --data /path/to/data.yaml --split test
```

### Performance Example (Detection)


| Metric                | Value                |
| --------------------- | -------------------- |
| **mAP50**             | 0.7532               |
| **mAP50-95**          | 0.5217               |
| **Precision**         | 0.7461               |
| **Recall**            | 0.6483               |
| **Inference Latency** | ~16ms (Apple M4 Pro) |


Evaluate a custom checkpoint with explicit backend type:

```bash
siin-trainer eval --backend custom --custom-backend-type ultralytics --checkpoint /path/to/custom.pt --data /path/to/data.yaml
```

Write metrics to a custom JSON path:

```bash
siin-trainer eval --backend rfdetr --data /path/to/dataset --model RFDETRMedium --output /path/to/eval_metrics.json
```

## Benchmark

Measure latency and throughput:

```bash
siin-trainer benchmark --backend ultralytics --checkpoint /path/to/best.pt --data /path/to/data.yaml --batch-size 1 --num-warmup 3 --num-iter 10
```

## 🌐 Live Web Demo

Run a modern, interactive dashboard to test your models with image upload, webcam support, and visualization controls.

```bash
siin-trainer demo
```

### Features

- **Real-time Inference**: Drag and drop images or use your camera.
- **Visualization Tuning**: Thin out box lines and toggle labels/confidence scores for crowded scenes.
- **Class Filtering**: Focus detections on specific categories.
- **Auto-scan**: Automatically detects all trained models in your `runs/` directory.

## Run full experiment from YAML

Run train + optional eval/benchmark:

```bash
siin-trainer run --config /path/to/experiment.yaml
```

Example:

```yaml
backend: ultralytics
data: /path/to/data.yaml
model: yolov8n
run_name: yolo-exp-001
train:
  epochs: 20
  img_size: 640
  batch: 16
run_eval: true
benchmark: true
benchmark_config:
  split: test
  batch_size: 1
  num_warmup: 3
  num_iter: 10
```

## Output layout

Artifacts are written under:

```text
runs/<backend>/<run_name_or_timestamp>/
```

Typical files:

- `train_artifacts.json`
- `eval_metrics.json`
- `benchmark.json`
- `config.yaml`

## Troubleshooting

- `siin-trainer: command not found`:
  - ensure virtualenv is activated
  - reinstall with `python -m pip install -e .`
- Python version error:
  - use Python `3.11` or `3.12` (`python --version`)
- RF-DETR eval/benchmark backend errors:
  - confirm compatible `rfdetr` package is installed
- Matplotlib/Ultralytics cache warnings:
  - CLI now auto-selects writable cache directories (`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `YOLO_CONFIG_DIR`)
  - first run may still take extra time while font/cache files are initialized