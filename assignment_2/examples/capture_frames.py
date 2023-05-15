import cv2
from scipy.io import loadmat


video_dir = "/externo/cv_data/NIR/Videos/"
matrix_dir = "/externo/cv_data/NIR/ObjectGT/"

matrix = matrix_dir + "MVI_1520_NIR_ObjectGT.mat"
mtx = loadmat(matrix)

# num_frames = len(mtx['structXML'][0])
# for frame_index in range(num_frames):
#     boxes = mtx["structXML"][0]['BB'][frame_index]
#     for box in boxes:
#         print(box)

# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(video_dir + "MVI_1520_NIR.avi")

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
    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is a frame
    ret, frame = vid_capture.read()
    if ret:
        boxes = mtx["structXML"][0]["BB"][frame_index]
        for box in boxes:
            print(box)
            x, y, w, h = box
            cv2.rectangle(
                frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2
            )

        cv2.imshow("Frame", frame)

        # 20 milliseconds
        key = cv2.waitKey(20)
        frame_index += 1

        if key == ord("q"):
            break
    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
