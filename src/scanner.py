import argparse
import pickle

from termcolor import colored

from backMarker import *
from laserPlane import *
from plateMarker import *
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

debug = args.debug

if debug:
    print(colored('- Debug mode activated', 'yellow'))

print("new camera matrix", new_camera_matrix)
print(colored("--- Parameters Loaded ---", "green"))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # undistort frame and crop
    frame = cv.undistort(frame, mtx, dist, None, new_camera_matrix)
    frame = frame[y:y + h, x:x + w]

    original = frame.copy()

    try:
        processedFrame = pre_process_frame(frame, debug=debug)
        contours, hierarchy = cv.findContours(processedFrame, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        rectangle = find_rectangle(contours)
        back_r, back_t, back_a, back_b, back_c = process_rectangle(frame, rectangle, mtx, dist,
                                                                   debug=debug)

        plate_elements = find_plate_elements(frame, contours, w, h)
        plate, plate_r, plate_t = process_plate(plate_elements, frame, original, mtx, dist, debug=debug)

        camera_to_plate = np.concatenate([
            np.concatenate([np.array(cv.transpose(cv.Rodrigues(plate_r)[0])),
                            np.array(- cv.transpose(cv.Rodrigues(plate_r)[0]) @ plate_t)], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0)

        center = cv.projectPoints(np.array([
            [0, 0, 0],
        ], dtype=np.float32), plate_r, plate_t, mtx, dist)[0][0][0]

        third = find_laser_plate_point(original.copy(), center, debug=debug)
        cv.drawMarker(frame, third, (255, 0, 0), cv.MARKER_CROSS, 10, 1)

        third_camera = np.array([third[0] - (w // 2), third[1] - (h // 2), mtx[0][0], 1])

        third_plate = camera_to_plate @ third_camera
        third_plate = [third_plate[0] / third_plate[3], third_plate[1] / third_plate[3],
                       third_plate[2] / third_plate[3]]

        origin = - np.transpose(cv.Rodrigues(plate_r)[0]) @ np.array(plate_t)
        origin = origin.transpose()[0]

        direction_vector = third_plate - origin
        t = - origin[2] / direction_vector[2]
        intersection_point = origin + (t * direction_vector)

        back_to_camera = np.concatenate([
            np.concatenate([cv.Rodrigues(back_r)[0], back_t], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0)
        first_back = [((-back_c - back_b * 0) / back_a), 0, 0, 1]
        second_back = [((-back_c - back_b * 230) / back_a), 230, 0, 1]
        first_camera = back_to_camera @ first_back
        second_camera = back_to_camera @ second_back
        first_plate = camera_to_plate @ first_camera
        first_plate = [first_plate[0] / first_plate[3], first_plate[1] / first_plate[3],
                       first_plate[2] / first_plate[3]]
        second_plate = camera_to_plate @ second_camera
        second_plate = [second_plate[0] / second_plate[3], second_plate[1] / second_plate[3],
                        second_plate[2] / second_plate[3]]

        plane = find_plane_equation(np.array(second_plate), np.array(first_plate), np.array(intersection_point))

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

        laser_points = detect_laser_points(original.copy(), plate, debug=debug)
        for i in laser_points:
            i_camera = np.array([i[0] - (w / 2), i[1] - (h / 2), mtx[0][0], 1])
            i_plate = camera_to_plate @ i_camera
            i_plate = [i_plate[0] / i_plate[3], i_plate[1] / i_plate[3],
                       i_plate[2] / i_plate[3]]

            point = find_plane_line_intersection(plane, origin, i_plate)

            test, _ = cv.projectPoints(np.array([
                point,
            ], dtype=np.float32), plate_r, plate_t, mtx, dist)

            output_file.write(f"{point[0]} {point[1]} {point[2]}\n")
            cv.drawMarker(frame, [round(m) for m in test[0][0]], (0, 0, 255), cv.MARKER_TILTED_CROSS, 5, 1)
            # cv.drawMarker(frame, [round(m) for m in i], (0, 255, 0), cv.MARKER_CROSS, 5, 1)

    except Exception as error:
        print(colored(f"ERROR: {error}", "red"))

    cv.putText(frame,
               f"{str(round(int(cap.get(cv.CAP_PROP_POS_FRAMES)) / int(cap.get(cv.CAP_PROP_FRAME_COUNT)) * 100, 2))}%",
               (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv.LINE_AA)

    output_file.flush()
    cv.imshow('scanner', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
output_file.close()

print("done")
