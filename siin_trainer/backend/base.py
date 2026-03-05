from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BackendError(RuntimeError):
    """Base exception raised by backend operations."""


class UnsupportedOperationError(BackendError):
    """Raised when a backend does not support an operation."""


class ModelBackend(ABC):
    """Common contract for train/eval/benchmark backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend unique identifier."""

    @abstractmethod
    def train(self, data_config: str, **kwargs: Any) -> dict[str, Any]:
        """Run training and return run metadata."""

    @abstractmethod
    def evaluate(
        self,
        checkpoint: str | None,
        data_config: str,
        split: str = "test",
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run evaluation and return metrics."""

    def benchmark(
        self,
        checkpoint: str | None,
        data_config: str,
        output_dir: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run benchmark and return latency/throughput metrics."""
        raise UnsupportedOperationError(f"Benchmark is not supported by backend '{self.name}'.")

    @staticmethod
    def ensure_output_dir(output_dir: str | None) -> Path | None:
        if not output_dir:
            return None
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        return target
