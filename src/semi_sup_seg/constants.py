from pathlib import Path

# Begin general filesystem constants. ##########################################

# This file is <project>/src/semi_sup_seg/constants.py.
PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR.parent / "data"

DEFAULT_RUNS_DIR = PROJECT_DIR / "logs" / "runs"

# End general filesystem constants. ############################################


DEFAULT_SEED = 20260912

MAX_PIXEL_INT_VALUE = 255  # Pixel intensities are uint8.
IMAGE_CHANNEL_COUNT = 3

IGNORE_LABEL_ID = 255
MAX_LABEL_COUNT = 256  # Labels are uint8.


# Begin Cityscapes constants. ##################################################

CITYSCAPES_DIR = DATA_DIR / "cityscapes"

CITYSCAPES_IMAGE_HEIGHT = 1024
CITYSCAPES_IMAGE_WIDTH = 2048

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
CITYSCAPES_VOID_CLASSES = {
    "unlabeled",
    "rectification border",
    "out of roi",
}
CITYSCAPES_EVAL_CLASSES = {
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
}
CITYSCAPES_PERSON_CLASSES = {
    "person",  # Person walking.
    "rider",  # Person riding a bicycle or motorcycle.
}

CITYSCAPES_CLASS_COLORS = {
    0: (0, 0, 0),  # Black.
    1: (0, 0, 0),  # Black.
    2: (0, 0, 0),  # Black.
    3: (0, 0, 0),  # Black.
    4: (0, 0, 0),  # Black.
    5: (111, 74, 0),  # Dark brown.
    6: (81, 0, 81),  # Dark purple.
    7: (128, 64, 128),  # Purple.
    8: (244, 35, 232),  # Magenta.
    9: (250, 170, 160),  # Pale salmon.
    10: (230, 150, 140),  # Salmon.
    11: (70, 70, 70),  # Dark gray.
    12: (102, 102, 156),  # Grayish blue.
    13: (190, 153, 153),  # Rosy brown.
    14: (180, 165, 180),  # Grayish lavender.
    15: (150, 100, 100),  # Muted reddish brown.
    16: (150, 120, 90),  # Muted brown.
    17: (153, 153, 153),  # Gray.
    18: (153, 153, 153),  # Gray.
    19: (250, 170, 30),  # Orange.
    20: (220, 220, 0),  # Yellow.
    21: (107, 142, 35),  # Olive drab.
    22: (152, 251, 152),  # Pale green.
    23: (70, 130, 180),  # Steel blue.
    24: (220, 20, 60),  # Crimson.
    25: (255, 0, 0),  # Red.
    26: (0, 0, 142),  # Dark blue.
    27: (0, 0, 70),  # Very dark blue.
    28: (0, 60, 100),  # Dark petrol blue.
    29: (0, 0, 90),  # Dark navy.
    30: (0, 0, 110),  # Navy.
    31: (0, 80, 100),  # Dark teal.
    32: (0, 0, 230),  # Blue.
    33: (119, 11, 32),  # Burgundy.
}

# End Cityscapes constants. ####################################################
