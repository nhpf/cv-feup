# ---- Training YOLO ----
# Extraction: Get frames + boxes from Singapore videos with 9:1 train-test
# Augmentation: Process image frames (histogram + sharpen)
# Train YOLO with extracted images but starting from yolov8m.pt (transfer learning)
# Now we have a nice trained .pt model for YOLO!

# ---- Training Classifier ----
# Prepare MARVEL dataset in local folder structure
# Train DeepSORT classifier with MARVEL? -> https://github.com/ZQPei/deep_sort_pytorch/issues/7 or https://github.com/abhyantrika/nanonets_object_tracking/
# Or should we use YOLOv8 as the classifier?

# TODO: Train DeepSORT tracker with Singapore?

# ---- Evaluation ----
# The evaluation script is like: https://github.com/yasarniyazoglu/YoloV5-and-DeepSort-Custom-Dataset/blob/main/track.py
# But it has to load trained models beforehand
# Embed DeepSORT into YOLO and load custom models

# Evaluate the model with mAP metrics using onshore and onboard Singapore videos as input:
# 1) Detection - Handled by YOLOv8
# 2) Association (Kalman Filter and IOU) - Handled by DeepSORT
# 3) Hungarian assignment - Handled by DeepSORT
