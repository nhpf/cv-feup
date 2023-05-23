import ultralytics

# Make sure my GPU is recognized (I use Arch btw)
ultralytics.checks(verbose=True)

# Ensure that the GPU will not run out of memory
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048"

# Get weights from YOLOv8 small (ideally it would be on medium)
detection_model = ultralytics.YOLO(model="yolov8s-cls.pt", task="classify")

# Train the model - transfer learning
detection_model.train(data="/externo/cv_data/marvel", epochs=20, imgsz=128)
