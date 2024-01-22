import numpy as np
import cv2 as cv
import pickle
import os

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cv.startWindowThread()

cap = cv.VideoCapture('../data/calibration.mov')
gray = None

i = 0
while cap.isOpened():
    ret, frame = cap.read()

    cap.set(cv.CAP_PROP_POS_FRAMES, i)

    if not ret:
        print("Video stream ended or in error")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(frame, (9, 6), corners2, ret) # corners2, ret)

    cv.imshow('calibrator', frame)
    if cv.waitKey(1) == ord('q'):
        break

    i += 60


cap.release()
cv.waitKey(1)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("rms: " + str(ret))

if not os.path.exists("../camera-parameters"):
    os.mkdir("../camera-parameters")

print(mtx)
file = open('../camera-parameters/camera-matrix', 'wb')
pickle.dump(mtx, file)
file.close()

print(dist)
file = open('../camera-parameters/camera-distortion', 'wb')
pickle.dump(dist, file)
file.close()

h, w = gray.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('../output/calibrationResult.png', dst)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))
