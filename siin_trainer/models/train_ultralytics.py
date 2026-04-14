"""Compatibility wrapper around the Ultralytics backend."""

from ..backend.registry import get_backend


def train_ultralytics_model(
    data_path: str,
    model_name: str = "yolov8n",
    epochs: int = 50,
    img_size: int = 640,
    batch=16,
    device="auto",
    cache="ram",
):
    """Train an Ultralytics YOLO model on a custom dataset."""
    backend = get_backend("ultralytics")
    return backend.train(
        data_config=data_path,
        model_name=model_name,
        epochs=epochs,
        img_size=img_size,
        batch=batch,
        device=device,
        cache=cache,
    )
