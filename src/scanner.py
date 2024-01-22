import math
import random
import sys

import numpy
import numpy as np
import cv2 as cv
import pickle
from utils import *
from marker import *
import os

marker_ref = create_virtual_marker()
cv.startWindowThread()
cv.namedWindow("scanner")
cv.namedWindow("debug")

#cv.createTrackbar("Min", "debug", 0, 255, lambda *args: None)
#cv.createTrackbar("Max", "debug", 0, 255, lambda *args: None)

#cv.setTrackbarPos("Min", "debug", 255)
#cv.setTrackbarPos("Max", "debug", 255)

if len(sys.argv) < 2:
    raise Exception("no input file, exiting")

cap = cv.VideoCapture(sys.argv[1])
gray = None

file = open('../camera-parameters/camera-matrix', 'rb')
mtx = pickle.load(file)
file.close()

file = open('../camera-parameters/camera-distortion', 'rb')
dist = pickle.load(file)
file.close()

print(mtx)
print(dist)

h, w = cap.read()[1].shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
x, y, w, h = roi

print(newcameramtx)
print("--- Parameters Loaded ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # undistort
    frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
    frame = frame[y:y + h, x:x + w]

    original = frame.copy()
    cv.imshow('original', original)

    cannied = pre_process_frame(frame)

    contours, hierarchy = cv.findContours(cannied, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # cv.drawContours(frame, contours, -1, (255, 0, 0), 5)

    rects = []
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    contours2 = []
    hierarchy2 = []
    # cv.drawContours(frame, contours, -1, (255,0,255), 5)
    for i, contour in enumerate(contours):

        '''
        else:
            cv.drawContours(frame, [contour], -1, (255,0,255), 5)
        '''

        # Approximate the contour to a polygon
        polygon = cv.approxPolyDP(contour, 0.1 * cv.arcLength(contour, True), True)

        # Check if the polygon has 4 sides and the aspect ratio is close to 1
        if len(polygon) == 4 and abs(
                1 - cv.contourArea(polygon) / (cv.boundingRect(polygon)[2] * cv.boundingRect(polygon)[3])) < 0.14:
            rects.append(polygon)
        else:
            # cv.drawContours(frame, [contour], -1, (0, 255, 255), 5)
            '''
            if hierarchy[0][i][2] != -1:
                cv.drawContours(frame, [contour], -1, (0, 0, 255), 5)
                continue
            '''
            hierarchy2.append(hierarchy[0][i])
            contours2.append(contour)

    # Draw rectangles
    '''
    for rect in rects:
        cv.drawContours(frame, [rect], -1, (0, 0, 255), 5)
    '''
    # result is dilated for marking the corners, not important
    # dst = cv.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    # frame[dst > 0.01 * dst.max()] = [0, 0, 255]

    # frame = cv.dilate(frame, None)
    # frame = frame[0:frame.shape[0]/2, 0:frame.] # frame.shape[0]/2:frame.shape[1]]

    # cv.drawContours(frame, gray, -1, (0, 255, 0), 3)

    disco = []

    for i, c in enumerate(contours2):
        if len(c) >= 5:
            ret = cv.fitEllipse(c)
            # if hierarchy[0][i][2] != -1:

            if 20 < ret[1][0] < 40 and 20 < ret[1][1] < 50 and 20 < ret[0][0] < w - 20 and h / 2 < ret[0][1] < h:
                # print(ret[1][0], ret[1][1])

                flag = True
                center = (round(ret[0][0]), round(ret[0][1]))

                for d in disco:
                    distance = math.sqrt((d[0] - center[0]) ** 2 + (d[1] - center[1]) ** 2)

                    if distance < 50:
                        flag = False

                if flag:
                    cv.ellipse(frame, ret, (0, 255, 0), thickness=5)
                    cv.drawMarker(frame, center, (0, 255, 0))
                    disco.append(center)

    '''
    
    filtered = []
    print(len(disco))
    for c in disco:
        for d in disco:
            cv.drawMarker(frame, (round(c[0][0]), round(c[0][1])), (255, 255, 0))
            cv.drawMarker(frame, (round(d[0][0]), round(d[0][1])), (255, 255, 0))
            c_distance = math.sqrt((c[0][0] - d[0][0])**2 + (c[0][1] - d[0][1])**2)
            if c != d and c not in filtered and c_distance > 500:
                filtered.append(c)

    print(len(filtered))
    disco = []
    '''

    if len(disco) > 0:
        center = my_ransac(frame, disco)

        if center is None:
            continue

        cv.ellipse(frame, center, (255, 0, 0), thickness=3)
        # cv.ellipse(frame, i, (255, 0, 0), thickness=3)
        # cv.drawMarker(frame, (round(center[0][0]), center(i[0][1])), (255, 0, 255))
        cv.drawMarker(frame, (round(center[0][0]), round(center[0][1])), (255, 0, 255))

        # cv.line(frame, center[0], (center[0][0], center[0][1] - 50), (100,100, 0), 5)

        disco = list(filter(lambda d: 20 < convert_to_polar(center, d)[0] < 310, disco))

        symbols = []

        for d in disco:
            color = get_point_color(original, d)
            if color is None:
                continue

            polars = convert_to_polar(center, d)
            # cv.putText(frame, color, d, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            # cv.putText(frame, f"{round(polars[1])}", d, cv.FONT_HERSHEY_SIMPLEX, 1,
            #           (0, 0, 0), 2, cv.LINE_AA)

            # index = polars[1] / 18
            symbols.append([polars[1], color, d, polars, -1])

        # symbols = symbols[:6]

        symbols.sort(key=lambda s: s[0], reverse=True)
        # print(list(map(lambda s: s[1], symbols)))

        t = []

        for i in range(len(symbols) - 1):
            t.append(symbols[i])
            if abs(symbols[i][0] - symbols[(i + 1) % len(symbols)][0]) % 360 > 30:
                t.append(None)

        symbols = t

        # print("".join(list(map(lambda s: s[1] if s is not None else '_', t))))
        cv.putText(frame, "".join(list(map(lambda s: s[1] if s is not None else '_', t))), (100, 100),
                   cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 0), 2, cv.LINE_AA)

        symbols += symbols

        # print(symbols)

        word = []

        '''
        for s in symbols:
            print(round(s[0]))
        '''

        for i in range(len(symbols) - 1):
            if symbols[i] is None or symbols[i + 1] is None:
                word = []
                continue
            # print(abs(symbols[i][0] - symbols[i + 1][0]))
            if abs(symbols[i][0] - symbols[i + 1][0]) < 30:
                word.append(symbols[i])
                if len(word) == 4:
                    break
            else:
                word = []

        if len(word) < 4:
            cv.putText(frame, "ERROR: word too short", (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1,
                       (255, 255, 0), 2, cv.LINE_AA)
            continue

        alphabet = 'YWMBMMCCCYWBMYWBYWBC'
        print(alphabet)
        alphabet = alphabet * 2

        # print("".join(map(lambda w: w[1], word)))
        offset = alphabet.find("".join(map(lambda w: w[1], word)))
        # print(offset)
        cv.putText(frame, str(offset), (100, 200), cv.FONT_HERSHEY_SIMPLEX, 1,
                   (255, 255, 0), 2, cv.LINE_AA)

        # symbols = [x for index, x in enumerate(symbols) if offset < index < offset + 19]
        print(offset)

        print("".join(list(map(lambda s: s[1] if s is not None else '_', symbols))))

        print("---")

        obj_points = []
        pln_points = []
        for i in range(19):
            if symbols[i] is None:
                break
            symbols[i][4] = (offset + i) % 20

            a = math.radians(18 * ((offset + i) % 20))

            pln_p = [
                (75 * np.cos(a)),
                (75 * np.sin(a)),
            ]

            obj_points.append(symbols[i][2])
            pln_points.append(pln_p)
            cv.putText(frame, f"{(offset + i) % 20}", symbols[i][2], cv.FONT_HERSHEY_SIMPLEX, 1,
                       (0, 0, 0), 2, cv.LINE_AA)

        H = cv.findHomography(np.array(obj_points), np.array(pln_points))

        transformed = cv.warpPerspective(original, H[0], (1000, 1000))

        # obj_points = list(map(lambda i: list(i).append(0), obj_points))

        obj_points = np.array(obj_points, dtype=np.float32)
        pln_points = np.array(pln_points, dtype=np.float32)

        obj_points = np.append(obj_points, np.zeros((len(obj_points), 1)), axis=1)

        _, r, t = cv.solvePnP(np.array(obj_points, dtype=np.float32), np.array(pln_points, dtype=np.float32),
                              mtx, dist, flags=cv.SOLVEPNP_IPPE)

        # r = cv.Rodrigues(r)
        print(t)

        g = cv.projectPoints(np.array(
            [
                0., 0., 10.
            ]
        ), r, t, mtx, dist)
        cv.drawMarker(frame, [round(i) for i in g[0][0][0]], (255, 0, 255), cv.MARKER_STAR)

        # test = [500, 500, 0]

        cv.imshow('transformed', transformed)

        '''
        print("--")
        print(alphabet)
        print("".join(list(map(lambda s: s[1], symbols))))
        print("--")
        '''

    cv.imshow('debug', cannied)
    cv.imshow('scanner', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.waitKey(1)
cv.destroyAllWindows()
cv.waitKey(1)
print("done")
