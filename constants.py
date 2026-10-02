from pathlib import Path

DEFAULT_SEED = 20260912
MAX_PIXEL_INT_VALUE = 255

DATA_DIR = Path("/home/stefan/hdd/projects/data")

CITYSCAPES_DIR = DATA_DIR / "cityscapes"

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
# Void classes are ignored during benchmark metric computation.
CITYSCAPES_VOID_CLASS_IDS = set(range(7))
# The 19 classes in the official Cityscapes benchmark.
CITYSCAPES_EVAL_CLASS_IDS = {
    7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
}

