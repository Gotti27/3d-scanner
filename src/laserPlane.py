import math

import cv2 as cv
import numpy as np


def find_laser_plate_point(frame, center):
    cropped = frame[
              round(center[0][0][1]) + 100:round(center[0][0][1]) + 200,
              round(center[0][0][0]) - 150:round(center[0][0][0]) + 150,
              :]  # , round(center[0][0][0]):]  # , 2]
    redChannel = cropped[:, :, 2]
    redChannel = cv.medianBlur(redChannel, 5)
    _, laser = cv.threshold(redChannel, 235, 255, cv.THRESH_BINARY)

    lines = cv.HoughLinesP(laser, 1, np.pi / 180, 50, None, 50, 20)

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
        cv.line(cropped, (longestLine[0], longestLine[1]), (longestLine[2], longestLine[3]), (0, 0, 255), 3)
        middle = (round((longestLine[0] + longestLine[2]) / 2), round((longestLine[1] + longestLine[3]) / 2))
        cv.drawMarker(cropped, middle, (0, 255, 0), cv.MARKER_CROSS, 15, 5)

    cv.imshow("third point", cropped)

    return [round(middle[0] + center[0][0][0] - 150), round(middle[1] + center[0][0][1] + 100)]
