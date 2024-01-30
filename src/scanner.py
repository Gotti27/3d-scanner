import argparse
import pickle

from marker import *
from src.backMarker import *
from src.laserPlane import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="inputFile", help="Input file path")
parser.add_argument("-o", "--output", dest="outputFile", help="Output file path")
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

if args.inputFile is None:
    raise Exception("no input file, exiting")
if args.outputFile is None:
    raise Exception("no output file, exiting")

cv.startWindowThread()
cv.namedWindow("scanner")

# Camera parameters loading
file = open('../camera-parameters/camera-matrix', 'rb')
mtx = pickle.load(file)
print("camera matrix", mtx)
file.close()

file = open('../camera-parameters/camera-distortion', 'rb')
dist = pickle.load(file)
print("camera distortion vector", dist)
file.close()

# opening outputfile
output_file = open(args.outputFile, "w")

# extracting video info from first frame and calculating newCameraMatrix
cap = cv.VideoCapture(args.inputFile)
h, w = cap.read()[1].shape[:2]
new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi

print("new camera matrix", new_camera_matrix)
print("--- Parameters Loaded ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # undistort frame and crop
    frame = cv.undistort(frame, mtx, dist, None, new_camera_matrix)
    frame = frame[y:y + h, x:x + w]

    original = frame.copy()

    canny = pre_process_frame(frame)
    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    rectangle = find_rectangle(frame, contours, w, h)
    back_r, back_t, back_a, back_b, back_c = process_rectangle(rectangle, frame, original, mtx, dist)

    plate_elements = find_plate_elements(frame, contours, w, h)
    plate, plate_r, plate_t = process_plate(plate_elements, frame, original, mtx, dist)

    camera_to_plate = np.concatenate([
        np.concatenate([np.array(cv.transpose(cv.Rodrigues(plate_r)[0])),
                        np.array(- cv.transpose(cv.Rodrigues(plate_r)[0]) @ plate_t)], axis=1),
        np.array([[0, 0, 0, 1]])
    ], axis=0)

    center = cv.projectPoints(np.array([
        [0, 0, 0],
    ], dtype=np.float32), plate_r, plate_t, mtx, dist)[0][0][0]

    third = find_laser_plate_point(original.copy(), center)
    third_camera = np.array([third[0], third[1], 1])
    third_camera = np.linalg.inv(mtx) @ third_camera
    third_camera = np.concatenate((third_camera, [1]))

    third_plate = camera_to_plate @ third_camera
    third_plate = [third_plate[0] / third_plate[3], third_plate[1] / third_plate[3],
                   third_plate[2] / third_plate[3]]

    origin = - np.transpose(cv.Rodrigues(plate_r)[0]) @ np.array(plate_t)
    origin = origin.transpose()[0]

    direction_vector = third_plate - origin
    t = - third_plate[2] / direction_vector[2]
    intersection_point = third_plate + t * direction_vector

    back_to_camera = np.concatenate([
        np.concatenate([cv.Rodrigues(back_r)[0], back_t], axis=1),
        np.array([[0, 0, 0, 1]])
    ], axis=0)
    first_back = [((-back_c - back_b * 0) / back_a), 0, 0, 1]
    second_back = [((-back_c - back_b * 230) / back_a), 230, 0, 1]
    first_camera = back_to_camera @ first_back
    second_camera = back_to_camera @ second_back
    first_plate = camera_to_plate @ first_camera
    first_plate = [first_plate[0] / first_plate[3], first_plate[1] / first_plate[3], first_plate[2] / first_plate[3]]
    second_plate = camera_to_plate @ second_camera
    second_plate = [second_plate[0] / second_plate[3], second_plate[1] / second_plate[3],
                    second_plate[2] / second_plate[3]]

    plane = find_plane_equation(second_plate, first_plate, intersection_point)

    o, _ = cv.projectPoints(np.array([
        intersection_point,
        [0, 0, 0],
        first_plate,
        second_plate
    ], dtype=np.float32), plate_r, plate_t, mtx, dist)

    cv.line(frame, [round(i) for i in o[0][0]], [round(i) for i in o[1][0]], (0, 0, 0), 5)
    cv.drawMarker(frame, [round(i) for i in o[0][0]], (0, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(frame, [round(i) for i in o[2][0]], (255, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(frame, [round(i) for i in o[3][0]], (255, 0, 0), cv.MARKER_CROSS, 30, 5)

    laser_points = detect_laser_points(original.copy(), plate)
    for i in laser_points:
        i_camera = np.array([i[0], i[1], 1])
        i_camera = np.linalg.inv(mtx) @ i_camera
        i_camera = np.concatenate((i_camera, [1]))
        i_plate = camera_to_plate @ i_camera
        i_plate = [i_plate[0] / i_plate[3], i_plate[1] / i_plate[3],
                   i_plate[2] / i_plate[3]]

        point = find_plane_line_intersection(plane, origin, i_plate)

        test, _ = cv.projectPoints(np.array([
            point,
        ], dtype=np.float32), plate_r, plate_t, mtx, dist)

        output_file.write(f"{point[0]} {point[1]} {point[2]}\n")
        cv.drawMarker(frame, [round(m) for m in test[0][0]], (0, 0, 255), cv.MARKER_TILTED_CROSS, 10, 3)

    cv.imshow('debug', canny)
    cv.imshow('scanner', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
output_file.close()

print("done")
