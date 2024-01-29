import pickle
import sys

from marker import *
from src.backMarker import find_rectangle, process_rectangle
from src.laserPlane import find_laser_plate_point
from utils import *

# marker_ref = create_virtual_marker()
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
    ret, frame = cap.read()
    if not ret:
        break

    # undistort and crop
    frame = cv.undistort(frame, mtx, dist, None, new_camera_matrix)
    frame = frame[y:y + h, x:x + w]

    original = frame.copy()

    canny = pre_process_frame(frame)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    rectangle = find_rectangle(frame, contours, w, h)
    back_r, back_t, back_a, back_b, back_c = process_rectangle(rectangle, frame, original, mtx, dist)

    plate = find_plate_elements(frame, contours, w, h)

    center, plate_r, plate_t = process_plate(plate, frame, original, mtx, dist)

    g, _ = cv.projectPoints(np.array([
        [0, 0, 0],
    ], dtype=np.float32), plate_r, plate_t, mtx, dist)

    third = find_laser_plate_point(original.copy(), g)

    third_camera = np.array([third[0], third[1], 1])
    third_camera = np.linalg.inv(mtx) @ third_camera
    third_camera = np.concatenate((third_camera, [1]))

    camera_to_plate = np.concatenate([
        np.concatenate([np.array(cv.transpose(cv.Rodrigues(plate_r)[0])),
                        np.array(- cv.transpose(cv.Rodrigues(plate_r)[0]) @ plate_t)], axis=1),
        np.array([[0, 0, 0, 1]])
    ], axis=0)

    third_plate = camera_to_plate @ third_camera
    third_plate = [third_plate[0] / third_plate[3], third_plate[1] / third_plate[3],
                   third_plate[2] / third_plate[3]]

    origin = - np.transpose(cv.Rodrigues(plate_r)[0]) @ np.array(plate_t)
    origin = origin.transpose()[0]

    direction_vector = third_plate - origin
    t = - third_plate[2] / direction_vector[2]
    intersection_point = third_plate + t * direction_vector

    o, _ = cv.projectPoints(np.array([
        intersection_point,
        [0, 0, 0]
    ], dtype=np.float32), plate_r, plate_t, mtx, dist)

    cv.line(frame, [round(i) for i in o[0][0]], [round(i) for i in o[1][0]], (0, 0, 0), 5)
    cv.drawMarker(frame, [round(i) for i in o[0][0]], (0, 255, 0), cv.MARKER_CROSS, 30, 5)

    cv.imshow('debug', canny)
    cv.imshow('scanner', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
print("done")
