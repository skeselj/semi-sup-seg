: """
Run some experiments. 
"""

PROFILE_RESOURCES=1 PROFILE_RESOURCES_TORCH=1 \
python ./train.py semi_sup south_5x5 &> \
    ../logs/stdout/south_5x5_semi_sup.txt


