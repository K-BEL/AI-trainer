from __future__ import annotations

from typing import Callable

from .base import BackendError, ModelBackend

BackendFactory = Callable[[], ModelBackend]


class BackendRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, BackendFactory] = {}

    def register(self, name: str, factory: BackendFactory) -> None:
        self._factories[name.lower()] = factory

    def create(self, name: str) -> ModelBackend:
        factory = self._factories.get(name.lower())
        if factory is None:
            available = ", ".join(sorted(self._factories.keys()))
            raise BackendError(f"Unknown backend '{name}'. Available backends: {available}")
        return factory()

    def available(self) -> list[str]:
        return sorted(self._factories.keys())


registry = BackendRegistry()


def _create_ultralytics_backend() -> ModelBackend:
    from .ultralytics_backend import UltralyticsBackend

    return UltralyticsBackend()


def _create_rfdetr_backend() -> ModelBackend:
    from .rfdetr_backend import RFDetrBackend

    return RFDetrBackend()


registry.register("ultralytics", _create_ultralytics_backend)
registry.register("rfdetr", _create_rfdetr_backend)


def get_backend(name: str) -> ModelBackend:
    return registry.create(name)
