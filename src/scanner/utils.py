import math
import random
from typing import Sequence

import cv2 as cv
import numpy as np


def pre_process_frame(frame, debug=False):
    """
    Frame pre-processor
    :param frame: the frame to be processed
    :param debug: debug mode flag
    :return: the pre-processed image
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # image to gray scale

    _, gray = cv.threshold(gray, 105, 255, cv.THRESH_BINARY_INV)  # inverted image threshold
    # dilate to ensure contours will be detectable
    gray = cv.morphologyEx(gray, cv.MORPH_DILATE, np.ones((3, 3), np.uint8))

    if debug:
        # draw all contours to the debug
        contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        copy = frame.copy()
        cv.drawContours(copy, contours, -1, (0, 255, 0), 5)
        cv.imshow('contours', copy)

    return gray


def render_ruler(frame):
    # just a debug function to render a ruler on the image
    cv.putText(frame, "|", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (200, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (300, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (400, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (500, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (600, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (700, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)
    cv.putText(frame, "|", (800, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv.LINE_AA)


def fit_ellipse_plate(frame, points, debug=False):
    """
    Ransac implementation to fit an ellipse from a set of points
    :param frame: the frame
    :param points: the list of points
    :param debug: debug mode flag
    :return: the best ellipse
    """
    if len(points) < 10:
        raise Exception("too few points")

    candidates = []
    for _ in range(0, 50):  # 50 rounds
        sampled = random.sample(points, 7)  # sample some points
        candidate = cv.fitEllipse(np.array(sampled))  # generate a candidate
        inliers = []  # inlier points list

        center_x, center_y = candidate[0]
        axis_1, axis_2 = candidate[1]
        angle = candidate[2]

        if math.isnan(axis_1) or math.isnan(axis_2):
            # not a valid candidate
            continue

        # approximate ellipse to polygon
        poly = cv.ellipse2Poly((round(center_x), round(center_y)), (round(axis_1 / 2), round(axis_2 / 2)),
                               round(angle), 0, 360, 1)

        for p in points:
            if abs(cv.pointPolygonTest(np.array(poly), p, True)) < 5:
                # the point is an inlier if its distance from the polygon is less than 5
                if debug:
                    cv.drawMarker(frame, np.array(p).astype(int), (255, 0, 255), cv.MARKER_TRIANGLE_UP)
                    cv.line(frame, np.array(p).astype(int), np.array(candidate[0]).astype(int), color=(0, 255, 255))
                inliers.append(p)

        candidates.append([candidate, len(inliers)])

    return max(candidates, key=lambda item: item[1])[0]  # extract the candidate with max votes


def are_all_leq(v1, v2):
    """
    Given two vectors, check if all elements of the first are leq wrt to the corresponding ones of the second vector
    :param v1: first vector
    :param v2: second vector
    :return: bool
    """
    if len(v1) != len(v2):
        raise Exception("dimension mismatch")

    for i in range(len(v1)):
        if v1[i] > v2[i]:
            return False
    return True


def get_point_color(frame, point):
    """
    Get the color of a given point coordinates or None if it cannot be matched
    :param frame:
    :param point:
    :return:
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # use hsv color space
    value = np.array(hsv[round(point[1])][round(point[0])])

    if are_all_leq(np.array([0, 0, 0]), value) and are_all_leq(value, np.array([180, 255, 90])):
        return "B"
    elif are_all_leq(np.array([0, 0, 180]), value) and are_all_leq(value, np.array([180, 99, 255])):
        return "W"
    elif are_all_leq(np.array([10, 100, 100]), value) and are_all_leq(value, np.array([40, 255, 255])):
        return "Y"
    elif are_all_leq(np.array([140, 100, 100]), value) and are_all_leq(value, np.array([170, 255, 255])):
        return "M"
    elif are_all_leq(np.array([80, 100, 100]), value) and are_all_leq(value, np.array([180, 255, 200])):
        return "C"
    else:
        return None


def convert_to_polar(ellipse: tuple[Sequence[float], Sequence[int], float], point):
    """
    Given an ellipse and a point, calculate the polar coordinates wrt to the ellipse center
    :param ellipse: the ellipse
    :param point: the point
    :return: the polar coordinates of the point wrt to the ellipse center
    """
    vector = (point[0] - ellipse[0][0], point[1] - ellipse[0][1])  # vector from ellipse center to point
    radius = np.linalg.norm(vector)  # norm of that vector
    angle = math.atan2(vector[1], vector[0])  # angle retrieved using atan

    angle_degrees = math.degrees(angle)  # conversion from radians to degrees
    if angle_degrees < 0:
        angle_degrees += 360

    return radius, angle_degrees


def find_line_equation(x1, y1, x2, y2):
    """
    Find the equation of the line passing through two points
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return: the line params in the form a, b, c
    """
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
    """
    Find the equation of the point passing through three points
    :param point1:
    :param point2:
    :param point3:
    :return:
    """
    vector1 = np.array(point2) - np.array(point1)
    vector2 = np.array(point3) - np.array(point1)
    normal_vector = np.cross(vector1, vector2)

    k = - np.sum(point1 * normal_vector)
    return np.array([normal_vector[0], normal_vector[1], normal_vector[2], k])


def find_plane_line_intersection(plane, point1, point2):
    """
    Find intersection between a plane and the line passing through two points
    :param plane:
    :param point1:
    :param point2:
    :return:
    """
    direction = point2 - point1
    plane_norm = np.array([plane[0], plane[1], plane[2]])
    product = plane_norm @ direction
    if abs(product) > 1e-6:
        p_co = plane_norm * (-plane[3] / (plane_norm @ plane_norm))

        w = point1 - p_co
        fac = - (plane_norm @ w) / product
        return point1 + (direction * fac)

    return None
