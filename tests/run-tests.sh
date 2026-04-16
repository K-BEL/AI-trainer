#!/bin/bash
set -e

# --- Environment Setup ---
# Find the project root robustly regardless of where the script is called from
SCRIPT_PATH="${BASH_SOURCE[0]}"
if [ -z "$SCRIPT_PATH" ]; then
    SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$( cd -- "$( dirname -- "$SCRIPT_PATH" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT"

# Determine the python to use (prefer venv if it exists)
if [[ -f "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

echo "🚀 Starting End-to-End Pipeline Test..."
echo "🐍 Using Python: $PYTHON_EXEC"
echo "📂 Project Root: $PROJECT_ROOT"

# Change to project root for consistent pathing
cd "$PROJECT_ROOT"

DATA_PATH="tests/dataset_tiny/data.yaml"
MODEL="yolov8n"

# 1. Train model (Minimal 5 epochs for testing, higher res)
echo "📦 Step 1: Training..."
"$PYTHON_EXEC" -m siin_trainer.cli train-ultralytics \
    --data "$DATA_PATH" \
    --model "$MODEL" \
    --device "cpu" \
    --epochs 5 \
    --batch 2 \
    --img-size 640 \
    --workers 2

# 2. Evaluate model
echo "📊 Step 2: Evaluating..."
# Finding the latest trained model weights in the custom runs folder
BEST_MODEL=$(ls -t runs/ultralytics/run-*/weights/best.pt | head -n 1)

"$PYTHON_EXEC" -m siin_trainer.cli eval \
    --backend ultralytics \
    --checkpoint "$BEST_MODEL" \
    --data "$DATA_PATH"

# 3. Benchmark model
echo "⏱️ Step 3: Benchmarking..."
"$PYTHON_EXEC" -m siin_trainer.cli benchmark \
    --backend ultralytics \
    --checkpoint "$BEST_MODEL" \
    --data "$DATA_PATH" \
    --num-iter 5

# 4. Visualize dataset
echo "🖼️ Step 4: Visualizing..."
"$PYTHON_EXEC" -m siin_trainer.cli visualize-dataset \
    --dataset "tests/dataset" \
    --output "runs/test_visualization" \
    --num-samples 2

# 5. Verify Success and Artifacts
echo "🔍 Step 5: Verifying..."
if [[ -f "$BEST_MODEL" ]]; then
    echo "✅ Success: Model checkpoint found at $BEST_MODEL"
else
    echo "❌ Error: Model checkpoint missing."
    exit 1
fi

EVAL_DIR="runs/ultralytics/eval"
BENCH_DIR="runs/ultralytics/benchmark"
VIS_DIR="runs/test_visualization"

if [[ -f "$EVAL_DIR/eval_metrics.json" ]]; then
    echo "✅ Success: Evaluation metrics found."
else
    echo "❌ Error: eval_metrics.json missing in $EVAL_DIR."
    exit 1
fi

if [[ -f "$BENCH_DIR/benchmark.json" ]]; then
    echo "✅ Success: Benchmark report found."
else
    echo "❌ Error: benchmark.json missing in $BENCH_DIR."
    exit 1
fi

if [[ -d "$VIS_DIR" ]]; then
    echo "✅ Success: Visualization output directory found."
else
    echo "❌ Error: Visualization directory missing."
    exit 1
fi

echo "✨ All end-to-end tests passed successfully!"
