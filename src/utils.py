import math
import random
from typing import Sequence

import cv2 as cv
import numpy as np


def pre_process_frame(frame, debug=False):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    _, gray = cv.threshold(gray, 105, 255, cv.THRESH_BINARY_INV)
    gray = cv.morphologyEx(gray, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))

    if debug:
        contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        copy = frame.copy()
        cv.drawContours(copy, contours, -1, (0, 255, 0), 5)
        cv.imshow('contours', copy)

    return gray


def render_ruler(frame):
    cv.putText(frame, "|", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (600, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (700, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (800, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)


def fit_ellipse_plate(frame, points, debug=False):
    if len(points) < 7:
        raise Exception("too few points")

    candidates = []
    for _ in range(0, 100):
        sampled = random.sample(points, 7)
        candidate = cv.fitEllipse(np.array(sampled))
        inliers = []

        center_x, center_y = candidate[0]
        axis_1, axis_2 = candidate[1]
        angle = candidate[2]

        for p in points:
            if math.isnan(axis_1) or math.isnan(axis_2):
                continue

            poly = cv.ellipse2Poly((round(center_x), round(center_y)), (round(axis_1 / 2), round(axis_2 / 2)),
                                   round(angle), 0, 360, 1)

            if abs(cv.pointPolygonTest(np.array(poly), p, True)) < 5:  # TODO: lower this threshold
                # cv.drawMarker(frame, np.array(p).astype(int), (255, 0, 255), cv.MARKER_TRIANGLE_UP)
                # cv.line(frame, np.array(p).astype(int), np.array(candidate[0]).astype(int), color=(0, 255, 255))
                inliers.append(p)

        candidates.append([candidate, len(inliers)])

    return max(candidates, key=lambda item: item[1])[0]


def are_all_leq(v1, v2):
    if len(v1) != len(v2):
        raise Exception("dimension mismatch")

    for i in range(len(v1)):
        if v1[i] > v2[i]:
            return False
    return True


def get_point_color(frame, point):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    value = np.array(hsv[round(point[1])][round(point[0])])

    if are_all_leq(np.array([0, 0, 0]), value) and are_all_leq(value, np.array([180, 255, 60])):
        return "B"
    elif are_all_leq(np.array([0, 0, 180]), value) and are_all_leq(value, np.array([180, 40, 255])):
        return "W"
    elif are_all_leq(np.array([20, 100, 100]), value) and are_all_leq(value, np.array([40, 255, 255])):
        return "Y"
    elif are_all_leq(np.array([140, 100, 100]), value) and are_all_leq(value, np.array([170, 255, 255])):
        return "M"
    elif are_all_leq(np.array([80, 100, 100]), value) and are_all_leq(value, np.array([110, 255, 255])):
        return "C"
    else:
        return None  # "{}, {}, {}".format(value[0], value[1], value[2])


def convert_to_polar(ellipse: tuple[Sequence[float], Sequence[int], float], point):
    vector = (point[0] - ellipse[0][0], point[1] - ellipse[0][1])

    radius = np.linalg.norm(vector)
    angle = math.atan2(vector[1], vector[0])

    angle_degrees = math.degrees(angle)
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


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


def find_plane_equation(point1, point2, point3):
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    normal_vector = np.cross(vector1, vector2)

    k = - np.sum(point1 * normal_vector)
    return np.array([normal_vector[0], normal_vector[1], normal_vector[2], k])


def find_plane_line_intersection(plane, point1, point2):
    direction = point2 - point1
    plane_norm = np.array([plane[0], plane[1], plane[2]])
    product = plane_norm @ direction
    if abs(product) > 1e-6:
        p_co = plane_norm * (-plane[3] / (plane_norm @ plane_norm))

        w = point1 - p_co
        fac = - (plane_norm @ w) / product
        return point1 + (direction * fac)

    return None
