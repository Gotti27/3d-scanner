import math

import cv2 as cv
import numpy as np

from utils import find_line_equation


def find_rectangle(contours):
    """
    Find rectangle marker from contours
    :param contours:
    :return:
    """
    rects = []

    for i, contour in enumerate(contours):
        # approximate contour to polygon
        polygon = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(contour)  # calculate polygon area

        if len(polygon) == 4 and area > 100000 and any([p[0][1] < 300 for p in polygon]):
            # if polygon is a large rectangle starting from the upper part of the image, it's a candidate
            rects.append(polygon)

    rects.sort(key=lambda x: cv.contourArea(x))
    return rects[0]  # pick the smallest one (i.e. the inner one)


def process_rectangle(frame, rectangle, mtx, dist, debug=False):
    """
    Process the rectangle and recover its pose and the laser line equation as a,b,c form
    :param frame: image frame
    :param rectangle: rectangle marker
    :param mtx: camera matrix
    :param dist: camera distortion parameters
    :param debug: debug flag
    :return: R, T, line_a, line_b, line_c
    """
    a, b, c, d = rectangle
    original = frame.copy()

    obj_points = [[0, 0, 0], [130, 0, 0], [130, 230, 0],
                  [0, 230, 0]]  # create object points knowing the real dimension of the marker
    pln_points = [a[0], b[0], c[0], d[0]]  # extract points from image rectangle

    # draw markers
    cv.drawMarker(frame, a[0], (255, 255, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, b[0], (0, 255, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, c[0], (255, 0, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, d[0], (0, 0, 255), cv.MARKER_TILTED_CROSS, 30, 5)

    # compute homography and transform the frame
    H = cv.findHomography(np.array(pln_points), np.array([[o[0], o[1]] for o in obj_points]))
    transformed = cv.warpPerspective(original, H[0], (1000, 1000))

    transformed = transformed[0:230, 0:130]  # crop around the marker

    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)  # compute pose

    redChannel = transformed[:, :, 2].copy()  # extract red channel (sufficient since not dealing with other colors)
    _, laser = cv.threshold(redChannel, 200, 255, cv.THRESH_BINARY)  # threshold red channel intensity
    lines = cv.HoughLinesP(laser, 1, np.pi / 360, 200, minLineLength=5)  # extract lines

    # pick the longest line (should be the most precise)
    longestLine = None
    if lines is not None and len(lines) > 0:
        longestLine = lines[0][0]
        for line in lines:
            length = math.dist((line[0, 0:1]), (line[0, 2:3]))
            longestLength = math.dist((longestLine[0:1]), (longestLine[2:3]))
            if longestLength < length:
                longestLine = line[0]

    if longestLine is None:
        raise Exception("laser line not found in back plane")

    # from the two points, compute the equation of the line
    line_a, line_b, line_c = find_line_equation(longestLine[0], longestLine[1], longestLine[2], longestLine[3])

    # some debug printing and projecting for validation
    if debug:
        cv.line(transformed, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), (0, 0, 255), 1)
        cv.drawMarker(transformed, (longestLine[0], longestLine[1]), (255, 0, 0), cv.MARKER_CROSS, 5, 1)
        cv.drawMarker(transformed, (longestLine[2], longestLine[3]), (0, 255, 0), cv.MARKER_CROSS, 5, 1)
        cv.imshow('back_plane homography', transformed)

    projected, _ = cv.projectPoints(np.array([
        [0, 0, 0],
        [130, 0, 0],
        [130, 230, 0],
        [0, 230, 0],
        [((-line_c - line_b * 0) / line_a), 0, 0],
        [((-line_c - line_b * 230) / line_a), 230, 0],
        [0, 0, 100]
    ], dtype=np.float32), r, t, mtx, dist)

    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[1][0]], (255, 0, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[2][0]], (0, 0, 255), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[3][0]], (0, 255, 0), 5)
    cv.line(frame, [round(i) for i in projected[0][0]], [round(i) for i in projected[6][0]], (255, 0, 255), 5)
    cv.line(frame, [round(i) for i in projected[4][0]], [round(i) for i in projected[5][0]], (0, 255, 0), 5)

    return r, t, line_a, line_b, line_c
