import argparse
import pickle

from termcolor import colored

from backMarker import *
from laserPlane import *
from plateMarker import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="inputFile", help="Input file path", required=True)
parser.add_argument("-o", "--output", dest="outputFile", help="Output file path", required=True)
parser.add_argument("-d", "--debug", dest="debug", help="Enable debug mode", action="store_true", default=False)
args = parser.parse_args()

if args.inputFile is None:
    raise Exception("no input file, exiting")
if args.outputFile is None:
    raise Exception("no output file, exiting")

cv.startWindowThread()
cv.namedWindow("scanner")

# Camera parameters loading
file = open('camera-parameters/camera-matrix', 'rb')
mtx = pickle.load(file)
print("camera matrix", mtx)
file.close()

file = open('camera-parameters/camera-distortion', 'rb')
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
    f = mtx[0][0]

    try:
        processedFrame = pre_process_frame(frame, debug=debug)  # preprocess frame
        contours, hierarchy = cv.findContours(processedFrame, cv.RETR_TREE,
                                              cv.CHAIN_APPROX_NONE)  # extract contours from preprocessed frame

        rectangle = find_rectangle(contours)  # find the backMarker rectangle
        # process the rectangle to recover its pose (r,t) and the parameters a,b,c of the laser line passing through
        # the back marker
        back_r, back_t, back_a, back_b, back_c = process_rectangle(frame, rectangle, mtx, dist,
                                                                   debug=debug)

        plate_elements = find_plate_elements(frame, contours)  # find the elements of the plate marker
        # process the elements of the plate marker to recover the plate marker ellipse and the marker pose (rt)
        plate, plate_r, plate_t = process_plate(plate_elements, frame, original, mtx, dist, debug=debug)

        # creating the transformation to map points from the camera reference system to the plate marker reference
        # system
        camera_to_plate = np.concatenate([
            np.concatenate([np.array(cv.transpose(cv.Rodrigues(plate_r)[0])),
                            np.array(- cv.transpose(cv.Rodrigues(plate_r)[0]) @ plate_t)], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0)

        # project the plate center on the image
        plate_center = cv.projectPoints(np.array([
            [0, 0, 0],
        ], dtype=np.float32), plate_r, plate_t, mtx, dist)[0][0][0]

        # creating the transformation to map points from the backMarker reference system to the camera reference system
        back_to_camera = np.concatenate([
            np.concatenate([cv.Rodrigues(back_r)[0], back_t], axis=1),
            np.array([[0, 0, 0, 1]])
        ], axis=0)

        '''
            the 3d laser plane can be recover from three points in the the 3d space:
            two points are given by the intersections of the laser line passing through the marker with the marker axes
            the third will be found on the plate (we know z = 0) 
        '''

        # find the first point in the back marker by imposing y = 0, z = 0 and put it in homogeneous coordinates
        first_back = [((-back_c - back_b * 0) / back_a), 0, 0,
                      1]
        # find the first point in the back marker by imposing y = 230, z = 0 and put it in homogeneous coordinates
        second_back = [((-back_c - back_b * 230) / back_a), 230, 0,
                       1]

        # map the two points to the camera projective space
        first_camera = back_to_camera @ first_back
        second_camera = back_to_camera @ second_back

        # map the two points to the plate reference system
        first_plate = camera_to_plate @ first_camera
        first_plate = [first_plate[0] / first_plate[3], first_plate[1] / first_plate[3],
                       first_plate[2] / first_plate[3]]

        second_plate = camera_to_plate @ second_camera
        second_plate = [second_plate[0] / second_plate[3], second_plate[1] / second_plate[3],
                        second_plate[2] / second_plate[3]]

        '''
            the third point can be found by intersecting the plate plane z = 0 with the ray passing through the camera 
            origin wrt the plate and a third point of the laser detected on the plate, mapped first in camera 
            reference system and then in plate coordinates
        '''
        # find a third laser point on the plate image
        third = find_laser_plate_point(original.copy(), plate_center, debug=debug)
        cv.drawMarker(frame, third, (255, 0, 0), cv.MARKER_CROSS, 10, 1)

        # map the third image point to the camera reference system
        third_camera = np.array([third[0] - (w // 2), third[1] - (h // 2), f, 1])
        third_plate = camera_to_plate @ third_camera  # map third point to plate reference system
        third_plate = [third_plate[0] / third_plate[3], third_plate[1] / third_plate[3],
                       third_plate[2] / third_plate[3]]  # extract 3d from homogeneous coordinates

        # compute the camera origin position with respect to the plate
        camera_origin = - np.transpose(cv.Rodrigues(plate_r)[0]) @ np.array(plate_t)
        camera_origin = camera_origin.transpose()[0]

        # compute the 3d line passing through the camera origin and the third point wrt the plate
        direction_vector = third_plate - camera_origin
        t = - camera_origin[2] / direction_vector[2]
        intersection_point = camera_origin + (t * direction_vector)
        # find the intersection between the line and the plate plane z = 0

        # compute the plane from the three points
        plane = find_plane_equation(np.array(second_plate), np.array(first_plate), np.array(intersection_point))

        projected, _ = cv.projectPoints(
            np.array([  # project back to the image the three points and plate origin to check visual correctness
                intersection_point,
                [0, 0, 0],
                first_plate,
                second_plate
            ], dtype=np.float32), plate_r, plate_t, mtx, dist)

        cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[1][0]], (0, 0, 0), 5)
        cv.drawMarker(frame, [round(i) for i in projected[0][0]], (0, 255, 0), cv.MARKER_CROSS, 30, 1)
        cv.drawMarker(frame, [round(i) for i in projected[2][0]], (255, 255, 0), cv.MARKER_CROSS, 30, 5)
        cv.drawMarker(frame, [round(i) for i in projected[3][0]], (255, 0, 0), cv.MARKER_CROSS, 30, 5)

        laser_points = detect_laser_points(original.copy(), plate,
                                           debug=debug)  # detect the laser points on the plate image

        for lp in laser_points:
            # map the laser point from the image to camera projective space
            i_camera = np.array([lp[0] - (w / 2), lp[1] - (h / 2), f, 1])
            i_plate = camera_to_plate @ i_camera  # map the point from camera reference system to plate coordinates
            i_plate = [i_plate[0] / i_plate[3], i_plate[1] / i_plate[3],
                       i_plate[2] / i_plate[3]]  # extract from homogeneous coordinates

            # compute the object point as the intersection between the line passing through the camera origin and the
            # laser point on the image with the 3d laser plane
            object_point = find_plane_line_intersection(plane, camera_origin, i_plate)

            if not (-40 < object_point[0] < 40 and -40 < object_point[1] < 40 and object_point[2] >= 0):
                continue

            # output the object point
            output_file.write(f"{object_point[0]} {object_point[1]} {object_point[2]}\n")
            cv.drawMarker(frame, lp, (0, 0, 255), cv.MARKER_TILTED_CROSS, 5, 1)

    except Exception as error:
        print(colored(f"ERROR: {error}", "red"))

    cv.putText(frame,
               f"{str(round(int(cap.get(cv.CAP_PROP_POS_FRAMES)) / int(cap.get(cv.CAP_PROP_FRAME_COUNT)) * 100, 2))}%",
               (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv.LINE_AA)

    output_file.flush()
    cv.imshow('scanner', frame)  # show the scanner window
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
output_file.close()

print(colored("-- done --", "green"))
