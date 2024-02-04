import math

import cv2 as cv
import numpy as np

from utils import fit_ellipse_plate, convert_to_polar, get_point_color


def find_plate_elements(frame: np.ndarray, contours):
    """
    Find the plate elements
    :param frame:
    :param contours: the contours list
    :return: the list of the candidate ellipses to be elements of the plate marker
    """
    plate_elements = []
    h, w, _ = frame.shape

    for i, c in enumerate(contours):
        if len(c) < 5:
            # not an ellipse for sure
            continue

        ellipse = cv.fitEllipse(c)  # fit ellipse in current contour

        if 20 < ellipse[1][0] < 60 and 20 < ellipse[1][1] < 60 and \
                20 < ellipse[0][0] < w - 20 and h / 2 < ellipse[0][1] < h:
            # good ellipse given thresholds on center position and axes length
            flag = True
            center = (round(ellipse[0][0]), round(ellipse[0][1]))

            for p in plate_elements:
                distance = math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

                if distance < 50:
                    # discard ellipse if the center is too close to another element
                    flag = False

            if flag:
                # draw ellipse and save center point
                cv.ellipse(frame, ellipse, (0, 255, 0), thickness=5)
                cv.drawMarker(frame, center, (0, 255, 0))
                plate_elements.append(center)

    return plate_elements


def process_plate(plate, frame, original, mtx, dist, debug=False):
    """
    Given the candidate elements, recover fiducial marker plate orientation and compute pose
    :param plate: marker plate elements
    :param frame:
    :param original: original frame
    :param mtx: camera matrix
    :param dist: camera distortion parameters
    :param debug: debug mode flag
    :return: center (the global ellipse marker), R, T
    """
    if len(plate) == 0:
        raise Exception("Cannot fit plate with no elements")

    center = fit_ellipse_plate(frame, plate, debug=debug)  # recover global ellipse marker from marker members

    if center is None:
        raise Exception("Cannot fit plate from the elements")

    if debug:
        cv.ellipse(frame, center, (255, 0, 0), thickness=3)

    # approximate marker ellipse to polygon
    poly = cv.ellipse2Poly((round(center[0][0]), round(center[0][1])),
                           (round(center[1][0] / 2), round(center[1][1] / 2)),
                           round(center[2]), 0, 360, 10)

    # discard outlier members
    plate = list(filter(lambda point: abs(cv.pointPolygonTest(poly, point, True)) < 5, plate))
    symbols = []

    for p in plate:
        color = get_point_color(original.copy(), p)  # for each marker element recover the color of its center
        if color is None:
            continue

        polars = convert_to_polar(center, p)  # convert the point coordinates in polar coordinates wrt the marker center
        # map the element into a symbol, composed by the polar angle, the color and the original position
        symbols.append([polars[1], color, p])

    symbols.sort(key=lambda s: s[0], reverse=True)  # sort the symbols according to the polar angle in reverse order
    symbols += symbols  # duplicate to make it 'circular'
    necklace = 'YWMBMMCCCYWBMYWBYWBC' * 2  # original fiducial marker necklace order

    elements_indexes = [[i, -1] for i in range(len(symbols) // 2)]  # assign at each element the index -1

    for i in range(len(symbols) // 2):
        votes = []
        # each element checks the next 3
        for j in range(4):
            if abs(symbols[(i + j)][0] - symbols[(i + j + 1)][0] + 360) % 360 > 25:
                # if the angle between two near elements is greater than 25, I assume there's a gap,
                # so its result is discarded
                votes = []
                break
            else:
                votes.append(symbols[i + j])

        if len(votes) == 0:
            continue

        # the substring is searched in the necklace to retrieve the element index
        temp_offset = necklace.find("".join(map(lambda v: v[1], votes)))
        if temp_offset == -1:
            # no match
            continue

        # the point assigns its index and the one of the next elements (possibly overriding them)
        for j in range(i, i + 5):
            elements_indexes[j % (len(symbols) // 2)] = [j % (len(symbols) // 2), temp_offset]
            temp_offset += 1
            temp_offset %= 20

    obj_points = []
    pln_points = []

    for i in range(20):
        if i >= len(elements_indexes):
            break
        if symbols[i] is None:
            continue

        offset = elements_indexes[i][1]  # retrieve index
        if offset == -1:
            continue

        # compute the corresponding object marker sibling using cosine and sine
        angle = math.radians(18 * offset)  # retrieve angle from index
        pln_p = [
            75 * np.cos(angle),
            75 * np.sin(angle)
        ]

        obj_points.append(pln_p)
        pln_points.append(symbols[i][2])

        cv.putText(frame, f"{symbols[i][1]}-{offset}", symbols[i][2], cv.FONT_HERSHEY_SIMPLEX, 0.7,
                   (200, 125, 200), 2, cv.LINE_AA)

    if len(obj_points) < 4 or len(pln_points) < 4:
        raise Exception("too few point to calculate plate pose")

    if debug:
        # find homography and show the transformation to debug
        H = cv.findHomography(np.array(pln_points),
                              np.array(
                                  list(map(lambda point: [(point[0] * 2) + 250, (point[1] * 2) + 250], obj_points))))
        transformed = cv.warpPerspective(original, H[0], (500, 500))
        cv.line(transformed, [250, 250], [75 * 2 + 250, 250], (255, 0, 0), 1)
        cv.line(transformed, [250, 250], [250, 75 * 2 + 250], (0, 255, 0), 1)
        cv.line(transformed, [250, 250], [-75 * 2 + 250, 250], (255, 255, 0), 1)
        cv.line(transformed, [250, 250], [250, -75 * 2 + 250], (0, 0, 255), 1)
        cv.drawMarker(transformed, [250, 250], (0, 0, 0), cv.MARKER_TILTED_CROSS, 15, thickness=1)
        cv.imshow('plate homography', transformed)

    obj_points = np.array(obj_points, dtype=np.float32)
    pln_points = np.array(pln_points, dtype=np.float32)
    # add third dimension to object points
    obj_points = np.append(obj_points, np.zeros((len(obj_points), 1)), axis=1)

    # estimate plate marker pose
    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    # project and render some debug points
    projected, _ = cv.projectPoints(np.array([
        [0, 0, 0],
        [75, 0, 0],
        [0, 75, 0],
        [0, -75, 0],
        [-75, 0, 0],
        [0, 0, 75],
    ], dtype=np.float32), r, t, mtx, dist)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[1][0]], (255, 0, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[2][0]], (0, 255, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[3][0]], (0, 0, 255), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[4][0]], (255, 255, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[5][0]], (255, 0, 255), 5)
    cv.drawMarker(frame, [round(i) for i in projected[0][0]], (255, 0, 255), cv.MARKER_STAR)

    return center, r, t
