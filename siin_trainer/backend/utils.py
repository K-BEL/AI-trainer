from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def now_run_name(prefix: str) -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def create_run_dir(backend_name: str, run_name: str | None = None, root: str = "runs") -> Path:
    name = run_name or now_run_name("run")
    run_dir = Path(root) / backend_name / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return str(value)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)


def list_images_from_data(data_config: str, split: str = "test", limit: int = 50) -> list[str]:
    import yaml

    data_path = Path(data_config)
    with data_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    split_path = data.get(split) or data.get("val") or data.get("train")
    if not split_path:
        return []

    split_file = Path(split_path)
    if not split_file.is_absolute():
        split_file = (data_path.parent / split_file).resolve()

    if split_file.is_file() and split_file.suffix.lower() == ".txt":
        with split_file.open("r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines[:limit]

    if split_file.is_dir():
        images: list[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            images.extend(str(p) for p in split_file.glob(ext))
        return images[:limit]

    return []
