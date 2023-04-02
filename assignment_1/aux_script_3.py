# Import dependencies
import numpy as np
import cv2
import glob

# Disclaimer: this code was partly inspired by the following sources
# A set of scripts that I wrote and made publicly available in 2020 on GitHub https://github.com/nhpf/opencv-classes
# The official OpenCV documentation https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# Calibration criteria. We perform 30 iterations with 0.001 subpixel precision
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(is_extrinsic, square_size_mm=22, width=7, height=4):
    """Takes images from IntrinsicCalibration directory and returns calibration parameters K, D"""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,3,0)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    # Scale the board to real size in meters
    if is_extrinsic:
        objp = objp * square_size_mm

    # Vetores para guardar os pontos nas imagens capturadads
    objpoints = []  # Pontos 3d da imagem
    imgpoints = []  # Pontos 2d no plano do tabuleiro

    # Iterate over the calibration images
    images = glob.glob("./IntrinsicCalibration/*.png")
    for fname in images:
        # Reads each image and converts it into grayscale
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found
        if ret:
            # Adds 3d points to objpoints list
            objpoints.append(objp)

            # Adjusts the detected chessboard corners according to precision criteria
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # Adds the adjusted corners to imgpoints list
            imgpoints.append(corners2)

            # This function draws the chessboard corners - used only for debugging
            # img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

    # Use the function calibrateCamera to generate correction matrix
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist


def undistort_and_get_roi(mtx, dist, fname):
    img = cv2.imread(fname)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # mapx, mapy are the correction factors that will be applied in each image to eliminate distortion
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    # Returns the undistorted frame
    corr = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Step I - Get region of interest
    x, y, w, h = roi
    corr = corr[y : y + h, x : x + w]

    # Limit the ROI to the inside of a white paper on the image
    # Binarize the image gray pixels between 160 and 255 are considered to be white
    gray = cv2.cvtColor(corr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    # Remove black gaps inside image (peanuts)
    kernel = np.ones((25, 25), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=4)

    cv2.imshow("binary roi", closing)

    # Get bounding rect for white pixels (paper)
    paper_points = cv2.findNonZero(closing)
    y, x, h, w = cv2.boundingRect(paper_points)

    # Add a 15px margin in each side to get a definive ROI 
    epsilon = 15
    corr = corr[x + epsilon : x+w-epsilon, y+epsilon:y+h-epsilon]
    cv2.imshow("roi", corr)

    cv2.waitKey(500000)

    return corr


def get_number_of_peanuts(mtx, dist):
    # Step II - Calculate the number of M&M per color for all images provided

    # Iterate over the calibration images
    images = glob.glob("./WhiteBackground/calib_img*.png")
    for fname in images:
        raw_img = undistort_and_get_roi(mtx, dist, fname)

        # blurred = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=3, sigmaY=3)
        # cv2.imshow('blurred'+fname, blurred)

        img = cv2.medianBlur(raw_img, ksize=15)
        # cv2.imshow("blurred" + fname, img)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Convert to binary
        threshold = 130
        max_value = 255

        thresh_type = cv2.THRESH_BINARY_INV
        # This function returns a tuple: (threshold, binary_image), hence the [1]
        inv_binary = cv2.threshold(gray, threshold, max_value, thresh_type)[1]

        # noise removal
        kernel = np.ones((3, 3), np.uint8)
        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # sure background area
        sure_bg = cv2.dilate(inv_binary, kernel, iterations=3)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(inv_binary, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0
        markers = cv2.watershed(img, markers)
        img[markers == -1] = [255, 0, 0]

        # kernel = np.ones((6, 6), 'uint8')
        # eroded = cv2.erode(inv_binary, kernel, iterations=3)
        # dilated = cv2.dilate(eroded, kernel, iterations=3)

        # n_labels, labels = cv2.connectedComponents(inv_binary)
        # print("Num objects: ", n_labels - 1)

        # # Map component labels to hue val
        # label_hue = np.uint8(179 * labels / np.max(labels))
        # blank_ch = 255 * np.ones_like(label_hue)
        # labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        #
        # # cvt to BGR for display
        # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        #
        # # set bg label to black
        # labeled_img[label_hue == 0] = 0
        #
        # # Display image using plt
        # plt.figure(figsize=(15, 15))
        #
        # plt.subplot(131)
        # plt.imshow(inv_binary, cmap="gray")
        # plt.title("Before Labeling")
        # plt.axis("off")
        #
        # plt.subplot(132)
        # plt.imshow(labeled_img)
        # plt.title("After labeling")
        # plt.axis("off")
        #
        # plt.subplot(133)
        # plt.imshow(gray, cmap="gray")
        # plt.title("Grayscale")
        # plt.axis("off")
        #
        # plt.show()

        cv2.imshow(fname, img)

    cv2.waitKey(5000000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # K, D = calibrate(is_extrinsic=False, square_size_mm=22, width=7, height=4)
    K = np.asarray([[1321.65, 0.0, 988.3], [0.0, 1324.446, 642.259], [0.0, 0.0, 1.0]])
    D = np.asarray([[-0.344, 0.09, 0.0, 0.0, -0.003]])

    # Question c)
    undistort_and_get_roi(
        K, D, "./WhiteBackground/calib_img 3.png"
    )

    # get_number_of_peanuts(K, D)

    # # Mostra a imagem original
    # cv2.imshow("Original", img)
    #
    # # Mostra a imagem corrigida
    # cv2.imshow("Corrigido", corr)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # Encerra o programa quando aperta a tecla 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # cv2.destroyAllWindows()
