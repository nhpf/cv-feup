import os
import cv2
from tqdm import tqdm
from glob import glob
from pathlib import Path
from random import shuffle
from scipy.io import loadmat

# Number N of seconds between each captured image
stride_seconds = 0.5

# Hard-coded paths
video_dir = lambda cat: f"/externo/cv_data/{cat}/Videos"
matrix_dir = lambda cat: f"/externo/cv_data/{cat}/ObjectGT"
output_dir = "/externo/cv_data/singapore"

# The resulting train/valid directories
train_images_dir = os.path.join(output_dir, "train", "images")
train_labels_dir = os.path.join(output_dir, "train", "labels")
valid_images_dir = os.path.join(output_dir, "validation", "images")
valid_labels_dir = os.path.join(output_dir, "validation", "labels")

# Will be used to collect images before splitting between train/valid
temp_dir = os.path.join(output_dir, "temp")

# Make sure that the directories defined above exist
for dirname in [
    train_images_dir,
    train_labels_dir,
    valid_images_dir,
    valid_labels_dir,
    temp_dir,
]:
    os.makedirs(dirname, exist_ok=True)

# Singapore data constants
capture_type = [
    "VIS_Onboard",
    "VIS_Onshore"
    # "NIR" -> We will not work with infra-red
]

# 1 Ferry | 2 Buoy | 3 Vessel/ship
# 4 Speed boat | 5 Boat | 6 Kayak
# 7 Sail boat | 8 Swimming person
# 9 Flying bird/plane | 10 Other
ignored_classes = [2, 8, 9, 10]

# For each category
for cap_type in capture_type:
    # Go through all video files
    for video_path in tqdm(glob(os.path.join(video_dir(cap_type), "*.avi")), desc=f"Processing {cap_type}"):
        video_name = Path(video_path).stem

        # Get annotations from the video
        matrix_file = os.path.join(matrix_dir(cap_type), f"{video_name}_ObjectGT.mat")

        # Ignore situations where we don't have the matrix file
        if not os.path.isfile(matrix_file):
            # print(f"Ignoring video: {video_path}")
            continue

        mtx = loadmat(matrix_file)

        # Create a video capture object
        vid_capture = cv2.VideoCapture(video_path)

        # Catch problems with video capture
        if not vid_capture.isOpened():
            raise Exception(f"Error opening the video file {video_path}")

        # Get number of frames per second
        fps = vid_capture.get(cv2.CAP_PROP_FPS)
        video_width = vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        frame_index = 0
        while vid_capture.isOpened():
            # Get frame from video
            ret, frame = vid_capture.read()
            if ret:
                # Get information about the objects present on frame
                boxes = mtx["structXML"][0]["BB"][frame_index]
                object_types = mtx["structXML"][0]["Object"][frame_index]

                # Write frame as a png file
                cv2.imwrite(
                    os.path.join(temp_dir, f"{video_name}_{frame_index}.png"), frame
                )

                # Ensure that a txt file wil also be created
                with open(
                    os.path.join(temp_dir, f"{video_name}_{frame_index}.txt"), "w"
                ) as labels_file:
                    # For each object
                    for obj_idx, obj_type in enumerate(object_types):
                        # Continue if there is no object in the image
                        if len(obj_type) < 1:
                            continue

                        obj_type = obj_type[0]

                        # Ignore irrelevant objects
                        if obj_type in ignored_classes:
                            continue

                        # Write object box coordinates in txt file
                        # The format is "class x_center y_center width height" normalized between 0 and 1
                        x, y, w, h = boxes[obj_idx]
                        x, w = max(0, x/video_width), w/video_width
                        y, h = max(0, y/video_height), h/video_height
                        labels_file.write(f"0 {x+w/2} {y+h/2} {w} {h}\n")

                # Jump to next position in video
                frame_index += int(fps * stride_seconds)
                vid_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            else:
                break

        # Release the video capture object
        vid_capture.release()
        # Destroy windows - only necessary after visualization
        cv2.destroyAllWindows()

# Now that we have all data on the temp directory, we will split it into train and validate

# Load all pairs of paths of images and labels to memory
pair_paths = []
for img_path in glob(os.path.join(temp_dir, "*.png")):
    txt_path = os.path.join(temp_dir, f"{Path(img_path).stem}.txt")
    pair_paths.append((Path(img_path), Path(txt_path)))

# Randomize the order of pairs of paths
shuffle(pair_paths)
print(f"\nWe have captured {len(pair_paths)} relevant frames\n")

# Split the data between train and validate with a 9:1 ratio
train_valid_split = 0.9
train_path_pairs = pair_paths[: int(len(pair_paths) * train_valid_split)]
valid_path_pairs = pair_paths[int(len(pair_paths) * train_valid_split) :]

# Move data from the temp directory to the assigned directory
for img_path, txt_path in train_path_pairs:
    img_path.rename(os.path.join(train_images_dir, img_path.name))
    txt_path.rename(os.path.join(train_labels_dir, txt_path.name))
for img_path, txt_path in valid_path_pairs:
    img_path.rename(os.path.join(valid_images_dir, img_path.name))
    txt_path.rename(os.path.join(valid_labels_dir, txt_path.name))
