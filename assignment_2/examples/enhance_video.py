import cv2
import numpy as np
from PIL import Image, ImageOps


# Hard-coded paths
video_dir = "/externo/cv_data/VIS_Onshore/Videos/"
matrix_dir = "/externo/cv_data/VIS_Onshore/ObjectGT/"
video_ref = "MVI_1474_VIS"


# Create two resizable preview windows
window1_name = "Original"
window2_name = "Enhanced"
cv2.namedWindow(window1_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(window2_name, cv2.WINDOW_NORMAL)


def histogram_correction(img):
    # Convert image to YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # Equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # Return image in BGR format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


def histogram_correction_pil(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    equalized = ImageOps.equalize(pil_img, mask=None)
    return cv2.cvtColor(np.array(equalized), cv2.COLOR_RGB2BGR)


# Thanks https://stackoverflow.com/a/55590133
def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


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
        enhanced = unsharp_mask(histogram_correction_pil(frame))
        cv2.imshow(window1_name, frame)
        cv2.imshow(window2_name, enhanced)

        # 20 milliseconds
        key = cv2.waitKey(20)
        if key == ord("q"):
            break

    else:
        break

# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()
