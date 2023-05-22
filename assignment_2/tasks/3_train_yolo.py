import os
import ultralytics

yaml_file = os.path.join(os.path.dirname(__file__), "..", "singapore.yaml")

# Make sure my GPU is recognized (I use Arch btw)
ultralytics.checks(verbose=True)

# Ensure that the GPU will not run out of memory
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

# Get weights from YOLOv8 nano (ideally it would be on medium)
detection_model = ultralytics.YOLO(model="yolov8n.pt", task="detect")

# Train the model - transfer learning
detection_model.train(data=yaml_file, epochs=20, imgsz=[1920, 1080], rect=True)
