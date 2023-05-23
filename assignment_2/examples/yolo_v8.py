import os
import cv2
from scipy.io import loadmat
from ultralytics import YOLO

# Load trained models
yolo_detection = YOLO(model="yolov8m.pt")
# yolo_detection = YOLO(model='/home/nick/repos/cv-feup/runs/detect/train5/weights/best.pt')
yolo_classification = YOLO(
    model="/home/nick/repos/cv-feup/assignment_2/classification_runs/train3/weights/last.pt"
)

# Hard-coded paths
video_dir = "/externo/cv_data/VIS_Onshore/Videos/"
matrix_dir = "/externo/cv_data/VIS_Onshore/ObjectGT/"
video_ref = "MVI_1474_VIS"

output_dir = os.path.join(os.path.dirname(__file__), "detection_results")

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

matrix = matrix_dir + f"{video_ref}_ObjectGT.mat"
mtx = loadmat(matrix)

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(video_dir + f"{video_ref}.avi")

if not vid_capture.isOpened():
    print("Error opening the video file")

# Read fps and frame count
else:
    # Get frame rate information
    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    print("Frames per second : ", fps, "FPS")

    # Get frame count
    frame_count = vid_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Frame count : ", frame_count)

frame_index = 0
while vid_capture.isOpened():
    ret, frame = vid_capture.read()
    if ret:
        detected = yolo_detection(frame)

        # cv2.imwrite(os.path.join(output_dir, f"{str(frame_index).zfill(4)}.png"), detected[0].plot())
        cv2.imshow("Detected by YOLO (standard)", detected[0].plot())
        frame_index += 1

        # 20 milliseconds
        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
