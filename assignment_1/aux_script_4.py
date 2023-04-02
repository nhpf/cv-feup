# Import dependencies
import numpy as np
import cv2
import glob

# Disclaimer: this code was partly inspired by the following sources
# A set of scripts that I wrote and made publicly available in 2020 on GitHub https://github.com/nhpf/opencv-classes
# The official OpenCV documentation https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

# Calibration criteria. We perform 30 iterations with 0.001 subpixel precision
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


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
    # corr = corr[0 : 0 + 760, 250 : 250 + 1120]  # Manually obtained in GIMP

    return corr


def get_color_masks(M, K, fname):
    # Get corrected image limited to its region of interest
    img = undistort_and_get_roi(M, K, fname)

    without_sp = cv2.medianBlur(img, ksize=21)
    without_sp = cv2.GaussianBlur(without_sp, ksize=(5, 5), sigmaX=10, sigmaY=10)

    color_ranges = {
        "brown": ([0, 50, 20], [25, 100, 140]),
        "yellow": ([15, 80, 80], [60, 255, 255]),
        "green": ([61, 100, 100], [90, 255, 255]),
        "blue": ([91, 180, 180], [150, 255, 255]),
        "red": ([151, 100, 100], [180, 255, 255]),
    }

    color_masks = {}

    hsv_img = cv2.cvtColor(without_sp, cv2.COLOR_BGR2HSV)

    for color_name in color_ranges.keys():
        filtered_img = cv2.inRange(
            hsv_img,
            np.array(color_ranges[color_name][0], np.uint8),
            np.array(color_ranges[color_name][1], np.uint8),
        )

        color_masks[color_name] = filtered_img

        # Deal with problematic brown color
        # if color_name == "brown":
        #     # Remove black gaps inside image
        #     kernel = np.ones((2, 2), np.uint8)
        #     without_artifacts = cv2.morphologyEx(
        #         filtered_img, cv2.MORPH_OPEN, kernel, iterations=6
        #     )
        #
        #     kernel = np.ones((8, 8), np.uint8)
        #     closed_img = cv2.morphologyEx(
        #         without_artifacts, cv2.MORPH_CLOSE, kernel, iterations=6
        #     )
        #
        #     color_masks[color_name] = closed_img
        # cv2.imshow(fname + color_name, filtered_img)
        # cv2.waitKey(10000000)

    # color_masks["brown"] = cv2.subtract(
    #     color_masks["brown"], cv2.bitwise_or(color_masks["red"], color_masks["yellow"])
    # )

    return color_masks


def get_number_of_peanuts_gray(M, K, fname):
    img = undistort_and_get_roi(M, K, fname=fname)

    color_masks = get_color_masks(M, K, fname=fname)

    num_peanuts = 0
    peanut_color_classes = {}

    for color_name in color_masks.keys():
        binary_img = color_masks[color_name]
        # sure background area
        # sure_bg = binary_img.copy()

        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(binary_img, cv2.DIST_L2, 3)
        # ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        #
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg, sure_fg)

        # # Marker labelling
        # ret, markers = cv2.connectedComponents(sure_fg)
        # # Add one to all labels so that sure background is not 0, but 1
        # markers = markers + 1
        # # Now, mark the region of unknown with zero
        # markers[unknown == 255] = 0

        # markers = cv2.watershed(original_img, markers)
        # original_img[markers == -1] = [255, 0, 0]

        # Find contours of that mask
        peanut_contours = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        peanut_contours = (
            peanut_contours[0] if len(peanut_contours) == 2 else peanut_contours[1]
        )

        correct_contours = []
        # For each contour
        for cnt in peanut_contours:
            cnt = cv2.convexHull(cnt, returnPoints=True)

            # If it has a reasonable size for a peanut
            if cv2.contourArea(cnt) > 400:
                peanut_area = cv2.contourArea(cnt)
                # Calculate its area
                num_peanuts += 1
                correct_contours.append(cnt)

                # Add one to corresponding color in dictionary
                if color_name in peanut_color_classes:
                    peanut_color_classes[color_name]["num_peanuts"] += 1
                    peanut_color_classes[color_name]["areas"].append(peanut_area)
                else:
                    peanut_color_classes[color_name] = {}
                    peanut_color_classes[color_name]["num_peanuts"] = 1
                    peanut_color_classes[color_name]["areas"] = [peanut_area]

        # Draw yellow contours in the original image
        for i in range(len(correct_contours)):
            cv2.drawContours(img, correct_contours, i, (0, 255, 255), 1)

    if num_peanuts > 0:
        return img, peanut_color_classes

    return img, peanut_color_classes


if __name__ == "__main__":
    # K, D = calibrate(is_extrinsic=False, square_size_mm=22, width=7, height=4)
    K = np.asarray([[1321.65, 0.0, 988.3], [0.0, 1324.446, 642.259], [0.0, 0.0, 1.0]])
    D = np.asarray([[-0.344, 0.09, 0.0, 0.0, -0.003]])

    """
    # 1. Define a ROI (region of interest) for the image;
    2. Calculate the number of M&M per color for all images provided;
    3. Determine the average area for peanuts in pixels, considering the image "calib_img 3";
        1. Show all peanuts that were detected.
        2. Discuss the limitations of the peanut detection method used (in 2 sentences).
    4. Determine the average area (and standard deviation) for peanuts in millimeters and grouped by color, considering images "calib_img_2" and "calib_img 3".
        1. Show the result in a table color vs area and standard deviation.
        2. Discuss the limitation of the method implemented (in 2 sentences).
    5. Provide some recommendations that the M&M factory should take into consideration to improve the performance of the quality assurance process based on image processing (e.g., imaging setup, calibration process and photometric effects), in 4 sentences.
    """

    img_paths = glob.glob("./GreyBackground/calib_img*.png")
    for img_idx, img_path in enumerate(img_paths):
        marked_img, classes = get_number_of_peanuts_gray(K, D, fname=img_path)
        cv2.imshow(img_path, marked_img)

        print(img_path, classes)

    cv2.waitKey(50000)
