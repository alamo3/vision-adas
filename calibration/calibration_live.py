"""
Code taken from : https://nikatsanka.github.io/camera-calibration-using-opencv-and-python.html

Learn more about camera calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""


import numpy as np
import cv2
import glob
import yaml

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

video_cap = cv2.VideoCapture(0)

found = 0
while found <=150:
    ret, img = video_cap.read() # Capture frame-by-frame

    cv2.imshow('img_raw', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,7), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print('found')
        obj_points.append(objp)   # Certainly, every loop objp is the same, in 3D.
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        img_points.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,7), corners2, ret)
        found += 1
        cv2.imshow('img', img)
        # if you want to save images with detected corners
        # uncomment following 2 lines and lines 5, 18 and 19
        # image_name = path + '/calibresult' + str(found) + '.png'
        # cv2.imwrite(image_name, img)

    cv2.waitKey(30)

print("Number of images used for calibration: ", found)

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()

# calibration
print('Calculating camera matrix')
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# transform the matrix and distortion coefficients to writable lists
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}

print('Saving camera matrix')
# and save it to a file
with open("calibration_matrix_live.yaml", "w") as f:
    yaml.dump(data, f)