import math

import cv2 as cv
import numpy as np

from utils import find_line_equation


def find_rectangle(contours):
    rects = []

    for i, contour in enumerate(contours):
        polygon = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        area = cv.contourArea(contour)

        if len(polygon) == 4 and area > 100000 and any([p[0][1] < 300 for p in polygon]):
            rects.append(polygon)

    rects.sort(key=lambda x: cv.contourArea(x))
    return rects[0]


def process_rectangle(frame, rectangle, mtx, dist, debug=False):
    a, b, c, d = rectangle

    obj_points = [[0, 0, 0], [130, 0, 0], [130, 230, 0], [0, 230, 0]]
    pln_points = [a[0], b[0], c[0], d[0]]

    cv.drawMarker(frame, a[0], (255, 255, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, b[0], (0, 255, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, c[0], (255, 0, 0), cv.MARKER_TILTED_CROSS, 30, 5)
    cv.drawMarker(frame, d[0], (0, 0, 255), cv.MARKER_TILTED_CROSS, 30, 5)

    H = cv.findHomography(np.array(pln_points), np.array([[o[0], o[1]] for o in obj_points]))
    transformed = cv.warpPerspective(frame.copy(), H[0], (1000, 1000))

    if debug:
        cv.line(transformed, [100 * 2, 100 * 2], [230 * 2, 100 * 2], (255, 0, 0), 5)
        cv.line(transformed, [100 * 2, 100 * 2], [100 * 2, 330 * 2], (0, 255, 0), 5)
        cv.line(transformed, [100 * 2, 100 * 2], [230 * 2, 330 * 2], (0, 0, 255), 5)

    transformed = transformed[0:230, 0:130]

    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    redChannel = transformed[:, :, 2].copy()
    _, laser = cv.threshold(redChannel, 200, 255, cv.THRESH_BINARY)

    lines = cv.HoughLinesP(laser, 1, np.pi / 180, 50, minLineLength=5)

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

    line_a, line_b, line_c = find_line_equation(longestLine[0], longestLine[1], longestLine[2], longestLine[3])

    if debug:
        cv.line(transformed, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), (0, 0, 255), 1)
        cv.drawMarker(transformed, (longestLine[0], longestLine[1]), (255, 0, 0), cv.MARKER_CROSS, 5, 1)
        cv.drawMarker(transformed, (longestLine[2], longestLine[3]), (0, 255, 0), cv.MARKER_CROSS, 5, 1)
        cv.imshow('back_plane', transformed)

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
