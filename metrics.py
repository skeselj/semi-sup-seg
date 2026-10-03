"""
Module to support evaluating semantic segmentation models.
"""

import torch

from data import LabelMetadata


def get_per_image_confusion_matrices(
    label: torch.Tensor,
    preds: torch.Tensor,
    num_classes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Given (B, H, W) label and preds, return the (B, N, N) confusion matrices.

    Element [b, i, j] is num. pixels in image b, with label i, predicted as j.
    """

    assert label.shape == preds.shape
    if not device:
        device = label.device

    batch_size = label.shape[0]
    flat_label = label.reshape(batch_size, -1).long()
    flat_preds = preds.reshape(batch_size, -1).long().to(device)

    batch_idxs = torch.arange(batch_size, device=device).unsqueeze(1)
    # Shape (B, H * W). Each value is b * N**2 + label * N + pred.
    flat_idxs = (
        batch_idxs * num_classes + flat_label
    ) * num_classes + flat_preds

    return torch.bincount(
        flat_idxs.flatten(), minlength=batch_size * num_classes**2
    ).reshape(batch_size, num_classes, num_classes)


def get_empty_confusion_matrix(
    label_metadata: LabelMetadata, device: torch.device | None = None
) -> torch.Tensor:
    """
    Get an all-zero confusion matrix.
    """

    # One extra row & column, for labels outside the classes.
    matrix_size = len(label_metadata.class_names) + 1
    return torch.zeros(
        (matrix_size, matrix_size), dtype=torch.long, device=device
    )


def get_batch_confusion_matrix(
    labels: torch.Tensor, preds: torch.Tensor, label_metadata: LabelMetadata
) -> torch.Tensor:
    """
    Get the (N + 1, N + 1) confusion matrix summed over (B, H, W) batches.

    N is the class count. Labels outside the classes, e.g. ignored ones,
    count in the extra row, which no metric reads.
    """

    class_count = len(label_metadata.class_names)
    labels = labels.where(labels < class_count, class_count)

    matrices = get_per_image_confusion_matrices(labels, preds, class_count + 1)
    return matrices.sum(dim=0)


def compute_public_benchmark_metrics(
    confusion: torch.Tensor, label_metadata: LabelMetadata
) -> dict[str, float]:
    """
    Compute benchmark metrics from confusion matrix summed over a dataset.
    """

    eval_ids = sorted(label_metadata.eval_class_ids)

    eval_rows = confusion[eval_ids].double()
    eval_block = eval_rows[:, eval_ids]

    intersection = eval_block.diagonal()
    union = eval_rows.sum(dim=1) + eval_block.sum(dim=0) - intersection
    ious = intersection / union  # NaN for a class never seen

    metrics = {
        "accuracy": (intersection.sum() / eval_rows.sum()).item(),
        "mean_iou": torch.nanmean(ious).item(),
    }
    for class_id, iou in zip(eval_ids, ious.tolist()):
        metrics[f"iou/{label_metadata.class_names[class_id]}"] = iou

    return metrics
