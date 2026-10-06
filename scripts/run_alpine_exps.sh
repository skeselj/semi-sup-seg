: """
Try using the ("alpine", "non-alpine") split.
"""

PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
python ./train.py "plain_supervised" 1st_alpine &> \
../logs/stdout/1st_alpine_plain_supervised.txt

PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
python ./train.py "augmented_supervised" 1st_alpine &> \
../logs/stdout/1st_alpine_augmented_supervised.txt

PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
python ./train.py "augmented_semisupervised" 1st_alpine &> \
../logs/stdout/1st_alpine_augmented_semisupervised.txt
