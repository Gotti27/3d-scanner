import math
import random
from typing import Sequence

import cv2 as cv
import numpy as np


def pre_process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # gray = cv.medianBlur(gray, 5)
    minThreshold = 255  # cv.getTrackbarPos('Min', 'debug')
    maxThreshold = 255  # cv.getTrackbarPos('Max', 'debug')

    # gray = cv.bitwise_not(gray)

    # gray = cv.medianBlur(gray, 5)
    test = gray
    # _, test = cv.threshold(gray, 180, 255, cv.THRESH_BINARY)
    # cannied = cv.Canny(gray, minThreshold, maxThreshold, L2gradient=True)
    # test = cv.morphologyEx(test, cv.MORPH_CLOSE, np.ones((23, 23), np.uint8))
    # test = cv.morphologyEx(test, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # cv.dilate(test, np.ones((9, 9), np.uint8), test, iterations=1)
    # cv.erode(test, np.ones((21, 21), np.uint8), test, iterations=1)

    cannied = cv.Canny(test, minThreshold, maxThreshold, L2gradient=True)
    # cv.dilate(cannied, np.ones((15, 15), np.uint8), test, iterations=1)
    # cv.erode(test, np.ones((15, 15), np.uint8), cannied, iterations=1)

    # test = cv.medianBlur(test, 3)
    _, test = cv.threshold(test, 90, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(test, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    hej = frame.copy()
    cv.drawContours(hej, contours, -1, (0, 255, 0), 5)
    cv.imshow('test', hej)

    cannied = cv.morphologyEx(cannied, cv.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # cannied = cv.morphologyEx(cannied, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    test = cv.morphologyEx(test, cv.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cannied
    # return test


def render_ruler(frame):
    cv.putText(frame, "|", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (600, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (700, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (800, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)


def my_ransac(frame, points):
    if len(points) < 7:
        return None
    candidates = []
    for _ in range(0, 100):
        sampled = random.sample(points, 7)
        candidate = cv.fitEllipse(np.array(sampled))
        inliers = []
        axis_1, axis_2 = candidate[1]

        for p in points:
            x1, y1 = p
            x2, y2 = candidate[0]

            # distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            if math.isnan(candidate[1][0]) or math.isnan(candidate[1][1]):
                continue

            e_p = cv.ellipse2Poly((round(x2), round(y2)), (round(candidate[1][0] / 2), round(candidate[1][1] / 2)),
                                  round(candidate[2]), 0, 360, 1)

            # cv.polylines(frame, [e_p], True, (0, 255, 255))

            # print(cv.pointPolygonTest(e_p, p, True))
            if abs(cv.pointPolygonTest(e_p, p, True)) < 5:
                cv.drawMarker(frame, np.array(p).astype(int), (255, 0, 255), cv.MARKER_TRIANGLE_UP)
                cv.line(frame, np.array(p).astype(int), np.array(candidate[0]).astype(int), color=(0, 255, 255))
                # cv.putText(frame, str(round(distance)), np.array(p).astype(int), cv.FONT_HERSHEY_SIMPLEX, 1,
                #           (255, 255, 0), 2, cv.LINE_AA)
                inliers.append(p)

        candidates.append([candidate, len(inliers)])

    # print(candidates)
    return max(candidates, key=lambda item: item[1])[0]

    # return cv.fitEllipse(np.array(points))


def get_point_color(frame, point):
    b, g, r = frame[round(point[1])][round(point[0])]
    if b < 80 and g < 80 and r < 80:
        return "B"
    elif 150 < b and 150 < g and 150 < r:
        return "W"
    elif 180 < g < 230 and 180 < r < 230:
        return "Y"
    elif 150 < r < 180:
        return "M"
    elif 150 < b < 200:
        return "C"
    else:
        return None  # "{}, {}, {}".format(b, g, r)


def convert_to_polar(ellipse: tuple[Sequence[float], Sequence[int], float], point):
    vector = (point[0] - ellipse[0][0], point[1] - ellipse[0][1])

    radius = np.linalg.norm(vector)
    angle = math.atan2(vector[1], vector[0])

    angle_degrees = math.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


def convert_to_cartesian(ellipse: tuple[Sequence[float], Sequence[int], float], radius, angle):
    radians = math.radians(angle)
    radius = np.linalg.norm(vector)
    angle = math.tan(vector[1], vector[0])


def find_line_equation(x1, y1, x2, y2):
    if x2 - x1 == 0:
        a = 1
        b = 0
        c = -x1
    else:
        m = (y2 - y1) / (x2 - x1)
        a = -m
        b = 1
        c = m * x1 - y1

    return a, b, c
