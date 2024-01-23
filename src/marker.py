import math

import cv2 as cv
import numpy as np

from src.utils import my_ransac, convert_to_polar, get_point_color


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


def find_plate_elements(frame: np.ndarray, contours, w, h):
    plate = []

    for i, c in enumerate(contours):
        if len(c) < 5:
            continue

        ellipse = cv.fitEllipse(c)
        # if hierarchy[0][i][2] != -1:

        if 20 < ellipse[1][0] < 40 and 20 < ellipse[1][1] < 50 and \
                20 < ellipse[0][0] < w - 20 and h / 2 < ellipse[0][1] < h:
            flag = True
            center = (round(ellipse[0][0]), round(ellipse[0][1]))

            for p in plate:
                distance = math.sqrt((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2)

                if distance < 50:
                    flag = False

            if flag:
                cv.ellipse(frame, ellipse, (0, 255, 0), thickness=5)
                cv.drawMarker(frame, center, (0, 255, 0))
                plate.append(center)

    return plate


def process_plate(plate, frame, original, mtx, dist):
    if len(plate) == 0:
        return

    center = my_ransac(frame, plate)

    if center is None:
        return

    cv.ellipse(frame, center, (255, 0, 0), thickness=3)
    # cv.ellipse(frame, i, (255, 0, 0), thickness=3)
    # cv.drawMarker(frame, (round(center[0][0]), center(i[0][1])), (255, 0, 255))
    cv.drawMarker(frame, (round(center[0][0]), round(center[0][1])), (255, 0, 255))

    # cv.line(frame, center[0], (center[0][0], center[0][1] - 50), (100,100, 0), 5)

    plate = list(filter(lambda d: 20 < convert_to_polar(center, d)[0] < 310, plate))

    symbols = []

    for p in plate:
        color = get_point_color(original, p)
        if color is None:
            continue

        polars = convert_to_polar(center, p)
        # cv.putText(frame, color, d, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
        # cv.putText(frame, f"{round(polars[1])}", d, cv.FONT_HERSHEY_SIMPLEX, 1,
        #           (0, 0, 0), 2, cv.LINE_AA)

        # index = polars[1] / 18
        symbols.append([polars[1], color, p, polars, -1])

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
        return

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
            (150 * np.cos(a)) + 500,  # (75 * np.cos(a)),
            (150 * np.sin(a)) + 500,  # (75 * np.sin(a)),
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
