"""Route Ultralytics TaskAlignedAssigner (TAL) through CPU on MPS.

PyTorch's MPS backend can return inconsistent element counts for boolean indexing on
`.expand()`'d tensors (see ultralytics issue #22971). Ultralytics upstream mitigates this
by running TAL on CPU when training on MPS; we apply the same approach at runtime so
older pinned ultralytics versions keep working.
"""

from __future__ import annotations

import torch
from lgg import logger

_PATCHED = False


def apply_task_aligned_assigner_mps_cpu_fallback() -> None:
    """Monkey-patch TaskAlignedAssigner.forward once so MPS training uses CPU for TAL."""
    global _PATCHED
    if _PATCHED:
        return

    from ultralytics.utils.tal import TaskAlignedAssigner

    if getattr(TaskAlignedAssigner, "_siin_mps_tal_cpu_fallback", False):
        _PATCHED = True
        return

    _original = TaskAlignedAssigner.forward

    @torch.no_grad()
    def _mps_safe_forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        device = gt_bboxes.device
        if device.type != "mps":
            return _original(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)

        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
        result = self._forward(*cpu_tensors)
        return tuple(t.to(device) for t in result)

    TaskAlignedAssigner.forward = _mps_safe_forward
    TaskAlignedAssigner._siin_mps_tal_cpu_fallback = True
    _PATCHED = True
    logger.info(
        "Applied Ultralytics MPS workaround: TaskAlignedAssigner runs on CPU (avoids MPS boolean-indexing bug)."
    )
