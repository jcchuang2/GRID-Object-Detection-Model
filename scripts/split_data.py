import splitfolders

splitfolders.ratio("data/images/image_data", output="data/images/image_data/image_split",
    seed=1337, ratio=(.8, .2), group_prefix=None, move=True)

