"""
Module to support working with sets of integer labels.
"""

import dataclasses

import torch

from semi_sup_seg.constants import (
    CITYSCAPES_CLASS_COLORS,
    CITYSCAPES_CLASS_NAMES,
    CITYSCAPES_EVAL_CLASSES,
    CITYSCAPES_PERSON_CLASSES,
    CITYSCAPES_VOID_CLASSES,
    IGNORE_LABEL_ID,
    MAX_LABEL_COUNT,
)

IGNORE_COLOR = (128, 128, 128)  # Gray.


@dataclasses.dataclass(frozen=True, kw_only=True)
class LabelMetadata:
    """
    Metadata that informs how to work with a set of labels.
    """

    # Raw label value (as in data files) --> label value. None means identity.
    raw_to_label: dict[int, int] | None = None

    class_names: dict[int, str]  # Label value --> class name.
    ignore_id: int = IGNORE_LABEL_ID  # Label value of ignored pixels.
    eval_class_ids: frozenset[int]  # Classes trained & evaluated on.
    class_colors: dict[int, tuple[int, int, int]]  # Label value --> RGB color.

    def get_train_label_map(self) -> torch.Tensor:
        """
        Get the int64 (MAX_LABEL_COUNT,) map: label value --> training label.
        """

        label_map = torch.full(
            (MAX_LABEL_COUNT,), self.ignore_id, dtype=torch.long
        )
        for class_id in self.eval_class_ids:
            label_map[class_id] = class_id

        return label_map

    def get_raw_train_label_map(self) -> torch.Tensor:
        """
        Get the int64 (MAX_LABEL_COUNT,) map: raw label --> training label.
        """

        train_label_map = self.get_train_label_map()
        if self.raw_to_label is None:
            return train_label_map

        raw_label_map = torch.full(
            (MAX_LABEL_COUNT,), self.ignore_id, dtype=torch.long
        )
        for raw_id, label_id in self.raw_to_label.items():
            raw_label_map[raw_id] = label_id

        return train_label_map[raw_label_map]


# Raw IDs of the Cityscapes classes trained & evaluated on.
_CITYSCAPES_EVAL_RAW_IDS = sorted(
    class_id
    for class_id, name in CITYSCAPES_CLASS_NAMES.items()
    if name in CITYSCAPES_EVAL_CLASSES
)

CITYSCAPES_LABEL_METADATA = LabelMetadata(
    class_names={
        train_id: CITYSCAPES_CLASS_NAMES[raw_id]
        for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
    },
    eval_class_ids=frozenset(range(len(_CITYSCAPES_EVAL_RAW_IDS))),
    class_colors={
        **{
            train_id: CITYSCAPES_CLASS_COLORS[raw_id]
            for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
        },
        IGNORE_LABEL_ID: IGNORE_COLOR,
    },
    raw_to_label={
        raw_id: train_id
        for train_id, raw_id in enumerate(_CITYSCAPES_EVAL_RAW_IDS)
    },
)

_PERSON_NEGATIVE_ID = 0
_PERSON_POSITIVE_ID = 1

# Person vs. background, from Cityscapes classes. Void classes are ignored.
CITYSCAPES_PERSON_LABEL_METADATA = LabelMetadata(
    class_names={
        _PERSON_NEGATIVE_ID: "background",
        _PERSON_POSITIVE_ID: "person",
    },
    eval_class_ids=frozenset({_PERSON_NEGATIVE_ID, _PERSON_POSITIVE_ID}),
    class_colors={
        _PERSON_NEGATIVE_ID: (0, 0, 0),
        _PERSON_POSITIVE_ID: CITYSCAPES_CLASS_COLORS[24],
        IGNORE_LABEL_ID: IGNORE_COLOR,
    },
    raw_to_label={
        class_id: (
            _PERSON_POSITIVE_ID
            if name in CITYSCAPES_PERSON_CLASSES
            else IGNORE_LABEL_ID
            if name in CITYSCAPES_VOID_CLASSES
            else _PERSON_NEGATIVE_ID
        )
        for class_id, name in CITYSCAPES_CLASS_NAMES.items()
    },
)
