import os
import shutil
from glob import glob
from pathlib import Path
from random import shuffle
import openml

# Constants
output_dir = "/externo/cv_data/marvel"
train_valid_split = 0.8

# The resulting train/valid directories
train_dir = os.path.join(output_dir, "train")
valid_dir = os.path.join(output_dir, "val")

# Will be used to collect images before splitting between train/valid
temp_dir = os.path.join(output_dir, "temp")

# Verify if all directories exist
if not all(
    [os.path.isdir(out_dir) for out_dir in [output_dir, train_dir, valid_dir, temp_dir]]
):
    raise Exception("Create the output directories before running this script!")

dataset_name = "BTS_Extended"

# Get dataset by name
dataset = openml.datasets.get_dataset(
    f"Meta_Album_{dataset_name}"
)  # download_data=True, download_all_files=True

# Get MARVEL data as a dataframe
marvel_df, _, _, _ = dataset.get_data(dataset_format="dataframe")

# Get the 26 vessel classes and store them in a map (enumeration)
vessel_classes = sorted(set(marvel_df["CATEGORY"]))
class_map = {cl: vessel_classes.index(cl) for cl in vessel_classes}
print(class_map)

# Create directories for each vessel class
for v_class in vessel_classes:
    for cur_dir in [
        os.path.join(base_dir, v_class) for base_dir in [train_dir, valid_dir, temp_dir]
    ]:
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)

# Compute directory where images are stored
img_dir = os.path.join(os.path.dirname(dataset.data_file), dataset_name, "images")

# For each image in the dataset
for idx, row in marvel_df.iterrows():
    img_file = os.path.join(img_dir, row["FILE_NAME"])
    img_class = row["CATEGORY"]

    temp_dir_class = os.path.join(temp_dir, img_class)
    temp_img_path = os.path.join(temp_dir_class, os.path.basename(img_file))

    # Copy file to temp classified directory
    shutil.copyfile(img_file, temp_img_path)


# For each vessel class
for v_class in vessel_classes:
    temp_dir_class = os.path.join(temp_dir, v_class)
    train_dir_class = os.path.join(train_dir, v_class)
    valid_dir_class = os.path.join(valid_dir, v_class)

    # Get a shuffled list of image paths for each vessel class
    image_paths = glob(os.path.join(temp_dir_class, "*"))
    shuffle(image_paths)

    # Split the data between train and validate with an 8:2 ratio
    train_image_paths = [
        Path(p) for p in image_paths[: int(len(image_paths) * train_valid_split)]
    ]
    valid_image_paths = [
        Path(p) for p in image_paths[int(len(image_paths) * train_valid_split) :]
    ]

    # Move data from the temp directory to the assigned directory
    for img_path in train_image_paths:
        img_path.rename(os.path.join(train_dir_class, img_path.name))
    for img_path in valid_image_paths:
        img_path.rename(os.path.join(valid_dir_class, img_path.name))
