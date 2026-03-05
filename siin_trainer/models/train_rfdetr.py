"""Compatibility wrapper around the RF-DETR backend."""

from ..backend.registry import get_backend


def train_rfdetr_model(
    data_path: str,
    model_name: str = "RFDETRMedium",
    epochs: int = 50,
    batch_size: int = 16,
    device: str = "cuda",
    resume: str = None,
):
    """Train an RF-DETR model on a custom dataset."""
    backend = get_backend("rfdetr")
    return backend.train(
        data_config=data_path,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        device=device,
        resume=resume,
    )
