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


def process_rectangle(rectangle, frame, original, mtx, dist):
    hej = frame.copy()
    a, b, c, d = rectangle

    obj_points = [[0, 0, 0], [130, 0, 0], [130, 230, 0], [0, 230, 0]]
    pln_points = [a[0], b[0], c[0], d[0]]

    cv.drawMarker(hej, a[0], (255, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, b[0], (0, 255, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, c[0], (255, 0, 0), cv.MARKER_CROSS, 30, 5)
    cv.drawMarker(hej, d[0], (0, 0, 255), cv.MARKER_CROSS, 30, 5)

    #

    H = cv.findHomography(np.array(pln_points), np.array([[o[0], o[1]] for o in obj_points]))

    _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    transformed = cv.warpPerspective(frame.copy(), H[0], (1000, 1000))

    '''
    cv.line(transformed, [100 * 2, 100 * 2], [230 * 2, 100 * 2], (255, 0, 0), 5)
    cv.line(transformed, [100 * 2, 100 * 2], [100 * 2, 330 * 2], (0, 255, 0), 5)
    cv.line(transformed, [100 * 2, 100 * 2], [230 * 2, 330 * 2], (0, 0, 255), 5)
    '''

    transformed = transformed[0:230, 0:130]
    redChannel = transformed[:, :, 2].copy()
    # redChannel = cv.medianBlur(redChannel, 5)
    _, laser = cv.threshold(redChannel, 200, 255, cv.THRESH_BINARY)
    # laser = cv.morphologyEx(laser, cv.MORPH_ERODE, np.ones((5, 5), np.uint8))
    # laser = cv.Canny(laser, 150, 200)
    # transformed = laser

    lines = cv.HoughLinesP(laser, 1, np.pi / 180, 50, None, 5)

    longestLine = None
    if lines is not None:
        for l in lines:
            if longestLine is None:
                longestLine = l[0]
                continue

            length = math.dist((l[0, 0:1]), (l[0, 2:3]))
            longestLength = math.dist((longestLine[0:1]), (longestLine[2:3]))
            if longestLength < length:
                longestLine = l[0]

    if longestLine is not None:
        cv.line(transformed, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), (0, 0, 255), 3)

    # cv.drawMarker(transformed, (longestLine[0], longestLine[1]), (255, 0, 0), cv.MARKER_CROSS, 15, 2)
    # cv.drawMarker(transformed, (longestLine[2], longestLine[3]), (0, 255, 0), cv.MARKER_CROSS, 15, 2)

    a, b, c = find_line_equation(longestLine[0], longestLine[1], longestLine[2], longestLine[3])

    cv.drawMarker(transformed, (round((-c - b * 0) / a), 0), (255, 255, 0), cv.MARKER_CROSS, 15, 2)
    cv.drawMarker(transformed, (round((-c - b * 230) / a), 230), (255, 255, 0), cv.MARKER_CROSS, 15, 2)
    cv.drawMarker(transformed, (0, 0), (0, 255, 0), cv.MARKER_CROSS, 15, 2)
    cv.drawMarker(transformed, (0, 230), (255, 0, 0), cv.MARKER_CROSS, 15, 2)
    cv.drawMarker(transformed, (130, 0), (0, 0, 255), cv.MARKER_CROSS, 15, 2)

    g, _ = cv.projectPoints(np.array([
        [0, 0, 0],
        [130, 0, 0],
        [130, 230, 0],
        [0, 230, 0],

        [((-c - b * 0) / a), 0, 0],
        [((-c - b * 230) / a), 230, 0],

        [0, 0, 100]
    ], dtype=np.float32), r, t, mtx, dist)
    # print("g (" + str(g[1][0]) + ")")
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[1][0]], (255, 0, 0), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[2][0]], (0, 0, 255), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[3][0]], (0, 255, 0), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[6][0]], (255, 0, 255), 5)
    # cv.drawMarker(frame, [round(i) for i in g[4][0]], (255, 255, 0), cv.MARKER_CROSS, 30, 5),
    # cv.drawMarker(frame, [round(i) for i in g[5][0]], (255, 255, 0), cv.MARKER_CROSS, 30, 5),

    cv.line(frame, [round(i) for i in g[4][0]], [round(i) for i in g[5][0]], (0, 255, 0), 5)
    '''
    cv.line(transformed, [500, 500], [500 + 300, 500], (255, 0, 0), 5)
    cv.line(transformed, [500, 500], [500, 500 + 300], (0, 255, 0), 5)
    cv.line(transformed, [500, 500], [500 - 300, 500], (255, 255, 0), 5)
    cv.line(transformed, [500, 500], [500, 500 - 300], (0, 0, 255), 5)
    cv.drawMarker(transformed, [500, 500], (0, 0, 0), cv.MARKER_STAR, thickness=5)
    # obj_points = list(map(lambda i: list(i).append(0), obj_points))

    obj_points = np.array(obj_points, dtype=np.float32)
    pln_points = np.array(pln_points, dtype=np.float32)

    pln_points = np.append(pln_points, np.zeros((len(pln_points), 1)), axis=1)


    '''

    # cv.drawContours(r, [rects[0]], -1, (255, 0, 0), 5)

    # cv.drawContours(r, rects[3], -1, (0, 255, 0), 5)

    cv.imshow('hej', hej)
    cv.imshow('t', transformed)
    return r, t, a, b, c
