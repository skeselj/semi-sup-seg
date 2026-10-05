"""
Module to support working with image labels.
"""

import dataclasses

import torch

from semi_sup_seg.constants import IGNORE_LABEL_ID, MAX_LABEL_COUNT

IGNORE_COLOR = (128, 128, 128)  # Gray.


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelMetadata:
    """
    Information for how to interpret labels of a certain type.
    """

    # To start, labels are loaded from file and passed through this map.
    raw_to_label: dict[int, int] | None = None
    # Labels not in this set are treated as ignore pixels.
    eval_class_ids: frozenset[int]
    # A pixel labeled with this value should be ignored.
    ignore_id: int = IGNORE_LABEL_ID

    # Label --> name.
    class_names: dict[int, str]
    # Label --> color.
    class_colors: dict[int, tuple[int, int, int]]

    def get_filter_to_eval_labels_map(self) -> torch.Tensor:
        """
        Return a label mapping that filters to `self.eval_class_ids`.
        """

        label_map = torch.full(
            (MAX_LABEL_COUNT,), self.ignore_id, dtype=torch.long
        )
        for class_id in self.eval_class_ids:
            label_map[class_id] = class_id

        return label_map

    def get_raw_to_train_label_map(self) -> torch.Tensor:
        """
        Return a mapping from raw label to label used for training.
        """

        train_label_map = self.get_filter_to_eval_labels_map()
        if self.raw_to_label is None:
            return train_label_map

        raw_label_map = torch.full(
            (MAX_LABEL_COUNT,), self.ignore_id, dtype=torch.long
        )
        for raw_id, label_id in self.raw_to_label.items():
            raw_label_map[raw_id] = label_id

        return train_label_map[raw_label_map]
