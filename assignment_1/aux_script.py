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
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow("img", img)
            cv2.waitKey(6000)

    # Use the function calibrateCamera to generate correction matrix
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    _, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist


if __name__ == "__main__":
    mtx, dist = calibrate(is_extrinsic=False, square_size_mm=22, width=7, height=4)

    img = cv2.imread("./WhiteBackground/calib_img 3.png")

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # mapx, mapy são os fatores de correção que serão aplicados em cada frame para eliminar a distorção
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    # Retorna o frame corrigido a partir dos parâmetros 'mapx' e 'mapy'
    corr = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Limita a imagem apenas à região de interesse
    x, y, w, h = roi
    corr = corr[y : y + h, x : x + w]

    # Mostra a imagem original
    cv2.imshow("Original", img)

    # Mostra a imagem corrigida
    cv2.imshow("Corrigido", corr)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Encerra o programa quando aperta a tecla 'q'
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # cv2.destroyAllWindows()
