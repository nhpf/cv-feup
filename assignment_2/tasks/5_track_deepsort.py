import cv2
from numpy import argmax
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

APPLY_CLASSIFICATION = False

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(
    "/externo/cv_data/VIS_Onboard/Videos/MVI_0799_VIS_OB.avi"
)
if not vid_capture.isOpened():
    print("Error opening the video file")
    exit()

# Create a video writer object
frame_width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid_capture.get(cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)

# Create a resizable window
window_name = "Vessel tracking"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Load the pre-trained YOLO detection model
yolo_detection = YOLO(
    "/home/nick/repos/cv-feup/assignment_2/detection_runs/v2/train3/weights/best.pt"
)
DETECTION_CONFIDENCE_THRESHOLD = 0.7

# Load the pre-trained YOLO classification model
yolo_classification = YOLO(
    "/home/nick/repos/cv-feup/assignment_2/classification_runs/train3/weights/last.pt"
)
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# Initialize the DeepSORT tracker
deepsort_tracker = DeepSort(max_age=50)

# Process all video frames
while True:
    start_time = datetime.now()
    ret, frame = vid_capture.read()

    if not ret:
        break

    # Perform object detection with YOLO
    detected = yolo_detection(frame)[0]

    # Initialize the list of bounding boxes and confidences
    detection_results = []

    # For each detected object
    for data in detected.boxes.data.tolist():
        # Get its associated probability
        confidence = data[4]

        # Do not consider detections with confidence below the threshold
        if float(confidence) < DETECTION_CONFIDENCE_THRESHOLD:
            continue

        # Get the box coordinates and the class id
        x_min, y_min, x_max, y_max = (
            int(data[0]),
            int(data[1]),
            int(data[2]),
            int(data[3]),
        )

        if APPLY_CLASSIFICATION:
            # Perform image classification with yolo
            classified = yolo_classification(frame[y_min:y_max, x_min:x_max])[0]
            vessel_classes = classified.names
            probabilities = classified.probs.tolist()

            # Do not consider vessels with confidence below the threshold
            if max(probabilities) < CLASSIFICATION_CONFIDENCE_THRESHOLD:
                continue

            # Determine the predicted vessel class
            vessel_class_id = argmax(probabilities)
            vessel_class = vessel_classes[vessel_class_id]
        else:
            vessel_class_id = int(data[5])

        # Add the bounding box (x, y, w, h), confidence and class id to the detection_results list
        detection_results.append(
            [(x_min, y_min, x_max - x_min, y_max - y_min), confidence, vessel_class_id]
        )

    # Now call the next iteration of DeepSORT
    tracks = deepsort_tracker.update_tracks(detection_results, frame=frame)

    # For each tracking result
    for track in tracks:
        # Ignore unconfirmed tracks
        if not track.is_confirmed():
            continue

        # Get the track id and its bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        x_min, y_min, x_max, y_max = (
            int(ltrb[0]),
            int(ltrb[1]),
            int(ltrb[2]),
            int(ltrb[3]),
        )

        # Draw the detected bounding box outline
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # Draw a green rectangle for the track ID
        cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 40, y_min), (0, 255, 0), -1)
        # Write the track ID
        cv2.putText(
            img=frame,
            text=str(track_id),
            org=(x_min + 5, y_min - 8),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=2,
        )

    # End time to compute the fps
    end_time = datetime.now()

    # Write FPS in the top left corner of the video
    fps = f"FPS: {1 / (end_time - start_time).total_seconds():.2f}"
    cv2.putText(frame, fps, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    # Live preview of the resulting frame
    # cv2.imshow(window_name, frame)

    # Write frame to mp4 file
    video_writer.write(frame)

    # Exit when "q" key is pressed
    if cv2.waitKey(1) == ord("q"):
        break

vid_capture.release()
video_writer.release()
cv2.destroyAllWindows()
