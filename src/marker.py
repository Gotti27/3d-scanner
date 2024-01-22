import math

import cv2 as cv
import numpy as np


def get_color(c: str):
    match c:
        case 'Y':
            return 0, 255, 255
        case 'W':
            return 255, 255, 255
        case 'M':
            return 255, 255, 0
        case 'C':
            return 255, 0, 0
        case 'B':
            return 0, 0, 0


def create_virtual_marker():
    center = [150, 150]
    # axis = [7.5 * 2, 7.5 * 2]
    angle = 0

    alphabet = ['Y', 'W', 'M', 'B', 'M', 'M', 'C', 'C', 'C', 'Y', 'W', 'B', 'M', 'Y', 'W', 'B', 'Y', 'W', "B", "C"]

    canvas = np.zeros((300, 300, 3), dtype="uint8")
    cv.ellipse(canvas, center, np.array([100, 100]), angle, 0, 360, (0, 255, 0), 5)
    cv.drawMarker(canvas, center, (0, 0, 255))

    points = []

    for i, s in enumerate(alphabet):
        a = math.radians(18 * i)

        cv.drawMarker(canvas, [
            round(150 + (100 * np.cos(a))),
            round(150 + (100 * np.sin(a))),
        ], get_color(s))

        cv.putText(canvas, str(i), [
            round(150 + (100 * np.cos(a))),
            round(150 + (100 * np.sin(a))),
        ], cv.FONT_HERSHEY_SIMPLEX, 1, get_color(s), 1, cv.LINE_AA)

        points.append((a, 100, s))

    cv.imshow("marker", canvas)

    return points
