from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from lgg import logger
from ultralytics import YOLO

from .base import ModelBackend
from .ultralytics_mps_workaround import apply_task_aligned_assigner_mps_cpu_fallback
from .utils import create_run_dir, list_images_from_data, to_jsonable, write_json


def _resolve_training_device(requested: str) -> str:
    """Map CLI/device hints to a device string Ultralytics can use."""
    raw = (requested or "auto").strip()
    if not raw:
        raw = "auto"
    r = raw.lower()
    if r == "auto":
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if r in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            logger.warning("CUDA was requested but is not available; using MPS instead.")
            return "mps"
        logger.warning("CUDA was requested but is not available; using CPU instead.")
        return "cpu"
    if r == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS was requested but is not available; using CPU instead.")
        return "cpu"
    if r in {f"{i}" for i in range(8)} or r.startswith("cuda:"):
        if torch.cuda.is_available():
            return raw
        if torch.backends.mps.is_available():
            logger.warning(
                f"GPU device {raw!r} was requested but CUDA is not available; using MPS instead."
            )
            return "mps"
        logger.warning(f"GPU device {raw!r} was requested but CUDA is not available; using CPU instead.")
        return "cpu"
    return raw


class UltralyticsBackend(ModelBackend):
    @property
    def name(self) -> str:
        return "ultralytics"

    def train(self, data_config: str, **kwargs: Any) -> dict[str, Any]:
        model_name = kwargs.get("model_name", "yolov8n")
        epochs = kwargs.get("epochs", 50)
        img_size = kwargs.get("img_size", 640)
        batch = kwargs.get("batch", 16)
        device = _resolve_training_device(kwargs.get("device", "auto"))
        cache = kwargs.get("cache", "ram")
        run_name = kwargs.get("run_name")
        runs_root = Path(kwargs.get("runs_root", Path.cwd() / "runs")).resolve()

        run_dir = create_run_dir(self.name, run_name=run_name, root=runs_root)
        if device == "mps":
            apply_task_aligned_assigner_mps_cpu_fallback()

        train_kw: dict[str, Any] = {
            "data": data_config,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch,
            "device": device,
            "cache": cache,
            "project": str(run_dir.parent),
            "name": run_dir.name,
            "exist_ok": True,
        }
        if kwargs.get("amp") is not None:
            train_kw["amp"] = kwargs["amp"]
        elif device == "mps":
            train_kw["amp"] = False

        model = YOLO(model_name)
        model.train(**train_kw)

        artifacts = {
            "backend": self.name,
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "weights" / "best.pt"),
            "last_checkpoint": str(run_dir / "weights" / "last.pt"),
        }
        write_json(artifacts, run_dir / "train_artifacts.json")
        return artifacts

    def evaluate(
        self,
        checkpoint: str | None,
        data_config: str,
        split: str = "test",
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_ref = checkpoint or kwargs.get("model_name", "yolov8n.pt")
        model = YOLO(model_ref)

        import yaml
        with Path(data_config).open("r", encoding="utf-8") as f:
            data_dict = yaml.safe_load(f) or {}

        actual_split = split
        if split not in data_dict:
            if "val" in data_dict:
                logger.warning(f"Split {split!r} not found in {data_config}; falling back to 'val'.")
                actual_split = "val"
            elif "train" in data_dict:
                logger.warning(f"Split {split!r} not found in {data_config}; falling back to 'train'.")
                actual_split = "train"

        metrics = model.val(data=data_config, split=actual_split)
        metrics_dict = {}
        if hasattr(metrics, "results_dict"):
            metrics_dict = to_jsonable(metrics.results_dict)
        elif isinstance(metrics, dict):
            metrics_dict = to_jsonable(metrics)
        else:
            metrics_dict = {"raw_metrics": to_jsonable(metrics)}

        eval_result = {
            "backend": self.name,
            "checkpoint": checkpoint,
            "data": data_config,
            "split": split,
            "metrics": metrics_dict,
        }

        target = self.ensure_output_dir(output_dir)
        if target:
            write_json(eval_result, target / "eval_metrics.json")
        return eval_result

    def benchmark(
        self,
        checkpoint: str | None,
        data_config: str,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_ref = checkpoint or kwargs.get("model_name", "yolov8n.pt")
        model = YOLO(model_ref)
        split = kwargs.get("split", "test")
        warmup = int(kwargs.get("num_warmup", 3))
        iterations = int(kwargs.get("num_iter", 10))
        batch_size = int(kwargs.get("batch_size", 1))

        images = list_images_from_data(data_config, split=split, limit=max(iterations, 10) * batch_size)
        if not images:
            result = {
                "backend": self.name,
                "checkpoint": checkpoint,
                "warning": "No images found for benchmark split.",
                "latency_ms_mean": None,
                "throughput_img_s": None,
            }
            target = self.ensure_output_dir(output_dir)
            if target:
                write_json(result, target / "benchmark.json")
            return result

        benchmark_batch = images[: batch_size * iterations]
        for _ in range(warmup):
            model.predict(source=benchmark_batch[:batch_size], verbose=False)

        latencies_ms: list[float] = []
        processed = 0
        cursor = 0
        for _ in range(iterations):
            batch = benchmark_batch[cursor : cursor + batch_size]
            if len(batch) < batch_size:
                batch = benchmark_batch[:batch_size]
            cursor += batch_size

            t0 = time.perf_counter()
            model.predict(source=batch, verbose=False)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)
            processed += len(batch)

        mean_latency = sum(latencies_ms) / len(latencies_ms)
        throughput = (processed / sum(latencies_ms)) * 1000 if latencies_ms else 0.0
        result = {
            "backend": self.name,
            "checkpoint": checkpoint,
            "iterations": iterations,
            "batch_size": batch_size,
            "latency_ms_mean": round(mean_latency, 3),
            "latency_ms_min": round(min(latencies_ms), 3),
            "latency_ms_max": round(max(latencies_ms), 3),
            "throughput_img_s": round(throughput, 3),
        }

        target = self.ensure_output_dir(output_dir)
        if target:
            write_json(result, target / "benchmark.json")
        return result
