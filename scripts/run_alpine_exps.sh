: """
Try using the ("alpine", "non-alpine") split.
"""

# Need to see some decent performance from this, maybe 0.5 IoU.
# PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
# python ./train.py augmented_supervised southern_split &> \
# ../logs/stdout/southern_split_augmented_supervised.txt

# Then will do semi-supervised, initializing from that checkpoint. 
PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
python ./train.py augmented_semisupervised southern_split &> \
../logs/stdout/southern_split_augmented_semisupervised.txt


