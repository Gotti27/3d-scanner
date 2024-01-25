import pickle
import sys

from marker import *
from src.backMarker import find_rectangle, process_rectangle
from utils import *

marker_ref = create_virtual_marker()
cv.startWindowThread()
cv.namedWindow("scanner")
# cv.namedWindow("debug")

if len(sys.argv) < 2:
    raise Exception("no input file, exiting")

# Camera parameters loading
file = open('../camera-parameters/camera-matrix', 'rb')
mtx = pickle.load(file)
print(mtx)
file.close()

file = open('../camera-parameters/camera-distortion', 'rb')
dist = pickle.load(file)
print(dist)
file.close()

cap = cv.VideoCapture(sys.argv[1])

h, w = cap.read()[1].shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi

print(new_camera_matrix)
print("--- Parameters Loaded ---")

while cap.isOpened():
    ellipse, frame = cap.read()
    if not ellipse:
        break

    # undistort and crop
    frame = cv.undistort(frame, mtx, dist, None, new_camera_matrix)
    frame = frame[y:y + h, x:x + w]

    original = frame.copy()

    canny = pre_process_frame(frame)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    rectangle = find_rectangle(frame, contours, w, h)
    process_rectangle(rectangle, frame, original, mtx, dist)

    plate = find_plate_elements(frame, contours, w, h)
    process_plate(plate, frame, original, mtx, dist)

    cv.imshow('debug', canny)
    cv.imshow('scanner', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
print("done")
