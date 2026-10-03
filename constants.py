from pathlib import Path

DEFAULT_SEED = 20260912
MAX_PIXEL_INT_VALUE = 255
IMAGE_CHANNEL_COUNT = 3

DATA_DIR = Path("/home/stefan/hdd/projects/data")

PROJECT_DIR = Path(__file__).resolve().parent

# Training runs: one directory per run, each with a checkpoint file.
DEFAULT_RUNS_DIR = PROJECT_DIR / "logs" / "runs"
DEFAULT_CHECKPOINT_FILE_NAME = "checkpoint.pt"

# BEGIN: Cityscapes-specific constants.

CITYSCAPES_DIR = DATA_DIR / "cityscapes"
CITYSCAPES_IMAGE_HEIGHT = 1024
CITYSCAPES_IMAGE_WIDTH = 2048

# Cityscapes classes: ID --> name.
CITYSCAPES_CLASS_NAMES = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
}
# The 19 classes in the official Cityscapes benchmark.
CITYSCAPES_EVAL_CLASS_IDS = {
    7,
    8,
    11,
    12,
    13,
    17,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    31,
    32,
    33,
}

# Cityscapes classes: ID --> official RGB color.
CITYSCAPES_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 0, 0),
    2: (0, 0, 0),
    3: (0, 0, 0),
    4: (0, 0, 0),
    5: (111, 74, 0),
    6: (81, 0, 81),
    7: (128, 64, 128),
    8: (244, 35, 232),
    9: (250, 170, 160),
    10: (230, 150, 140),
    11: (70, 70, 70),
    12: (102, 102, 156),
    13: (190, 153, 153),
    14: (180, 165, 180),
    15: (150, 100, 100),
    16: (150, 120, 90),
    17: (153, 153, 153),
    18: (153, 153, 153),
    19: (250, 170, 30),
    20: (220, 220, 0),
    21: (107, 142, 35),
    22: (152, 251, 152),
    23: (70, 130, 180),
    24: (220, 20, 60),
    25: (255, 0, 0),
    26: (0, 0, 142),
    27: (0, 0, 70),
    28: (0, 60, 100),
    29: (0, 0, 90),
    30: (0, 0, 110),
    31: (0, 80, 100),
    32: (0, 0, 230),
    33: (119, 11, 32),
}

# END: Cityscapes-specific constants.
