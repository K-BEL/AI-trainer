from __future__ import annotations

import inspect
import time
from typing import Any

from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from .base import BackendError, ModelBackend, UnsupportedOperationError
from .utils import create_run_dir, list_images_from_data, to_jsonable, write_json


RFDETR_MODEL_MAP = {
    "RFDETRMedium": RFDETRMedium,
    "RFDETRNano": RFDETRNano,
    "RFDETRSmall": RFDETRSmall,
    "RFDETRLarge": RFDETRLarge,
    "RFDETRBase": RFDETRBase,
}


def _call_supported(fn: Any, args_map: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    kwargs = {key: value for key, value in args_map.items() if key in sig.parameters}
    return fn(**kwargs)


class RFDetrBackend(ModelBackend):
    @property
    def name(self) -> str:
        return "rfdetr"

    def _build_model(self, model_name: str) -> Any:
        model_cls = RFDETR_MODEL_MAP.get(model_name)
        if not model_cls:
            raise BackendError(f"Unsupported RF-DETR model name: {model_name}")
        return model_cls()

    def train(self, data_config: str, **kwargs: Any) -> dict[str, Any]:
        model_name = kwargs.get("model_name", "RFDETRMedium")
        epochs = kwargs.get("epochs", 50)
        batch_size = kwargs.get("batch_size", 16)
        device = kwargs.get("device", "cuda")
        resume = kwargs.get("resume")
        run_name = kwargs.get("run_name")
        runs_root = kwargs.get("runs_root", "runs")

        run_dir = create_run_dir(self.name, run_name=run_name, root=runs_root)
        model = self._build_model(model_name)
        model.train(
            dataset_dir=data_config,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            resume=resume,
            num_workers=8,
            lr=1e-4,
            grad_accum_steps=4,
            wandb=True,
            project=str(run_dir.parent),
        )
        model.export(output_dir=str(run_dir), simplify=True, opset_version=12)

        artifacts = {
            "backend": self.name,
            "run_dir": str(run_dir),
            "model_name": model_name,
            "resume": resume,
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
        model_name = kwargs.get("model_name", "RFDETRMedium")
        model = self._build_model(model_name)

        if checkpoint and hasattr(model, "load"):
            try:
                _call_supported(model.load, {"checkpoint": checkpoint, "weights": checkpoint, "path": checkpoint})
            except Exception:
                pass

        eval_methods = ["evaluate", "val", "test"]
        eval_output: Any = None
        for method_name in eval_methods:
            method = getattr(model, method_name, None)
            if method is None:
                continue
            try:
                eval_output = _call_supported(
                    method,
                    {
                        "dataset_dir": data_config,
                        "data": data_config,
                        "split": split,
                        "checkpoint": checkpoint,
                    },
                )
                break
            except Exception:
                continue

        if eval_output is None:
            raise UnsupportedOperationError(
                "RF-DETR backend could not find a usable evaluation method. "
                "Please verify installed rfdetr version supports evaluate/val/test."
            )

        result = {
            "backend": self.name,
            "checkpoint": checkpoint,
            "data": data_config,
            "split": split,
            "metrics": to_jsonable(eval_output),
        }
        target = self.ensure_output_dir(output_dir)
        if target:
            write_json(result, target / "eval_metrics.json")
        return result

    def benchmark(
        self,
        checkpoint: str | None,
        data_config: str,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_name = kwargs.get("model_name", "RFDETRMedium")
        split = kwargs.get("split", "test")
        warmup = int(kwargs.get("num_warmup", 3))
        iterations = int(kwargs.get("num_iter", 10))
        batch_size = int(kwargs.get("batch_size", 1))
        model = self._build_model(model_name)

        if checkpoint and hasattr(model, "load"):
            try:
                _call_supported(model.load, {"checkpoint": checkpoint, "weights": checkpoint, "path": checkpoint})
            except Exception:
                pass

        infer_method = getattr(model, "predict", None) or getattr(model, "infer", None)
        if infer_method is None:
            raise UnsupportedOperationError("RF-DETR backend does not expose predict/infer for benchmarking.")

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
            _call_supported(infer_method, {"images": benchmark_batch[:batch_size], "source": benchmark_batch[:batch_size]})

        latencies_ms: list[float] = []
        processed = 0
        cursor = 0
        for _ in range(iterations):
            batch = benchmark_batch[cursor : cursor + batch_size]
            if len(batch) < batch_size:
                batch = benchmark_batch[:batch_size]
            cursor += batch_size

            t0 = time.perf_counter()
            _call_supported(infer_method, {"images": batch, "source": batch})
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
