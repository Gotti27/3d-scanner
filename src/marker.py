import math

import cv2 as cv
import numpy as np

from utils import fit_ellipse_plate, convert_to_polar, get_point_color


def find_plate_elements(frame: np.ndarray, contours, w, h):
    plate = []

    for i, c in enumerate(contours):
        if len(c) < 5:
            continue

        ellipse = cv.fitEllipse(c)

        if 20 < ellipse[1][0] < 60 and 20 < ellipse[1][1] < 60 and \
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


def process_plate(plate, frame, original, mtx, dist, debug=False):
    if len(plate) == 0:
        return

    center = fit_ellipse_plate(frame, plate)

    if center is None:
        raise Exception("Cannot fit plate from the elements")

    if debug:
        cv.ellipse(frame, center, (255, 0, 0), thickness=3)

    poly = cv.ellipse2Poly((round(center[0][0]), round(center[0][1])),
                           (round(center[1][0] / 2), round(center[1][1] / 2)),
                           round(center[2]), 0, 360, 10)

    plate = list(filter(lambda point: abs(cv.pointPolygonTest(poly, point, True)) < 5, plate))
    symbols = []

    for p in plate:
        color = get_point_color(original.copy(), p)
        if color is None:
            continue

        polars = convert_to_polar(center, p)
        symbols.append([polars[1], color, p, polars, -1])

    symbols.sort(key=lambda s: s[0], reverse=True)
    symbols += symbols

    necklace = 'YWMBMMCCCYWBMYWBYWBC' * 2

    total = [[i, -1] for i in range(len(symbols) // 2)]

    for i in range(len(symbols) // 2):
        votes = []
        for j in range(4):
            # print(abs(symbols[(i + j) % len(symbols)][0] - symbols[(i + j + 1) % len(symbols)][0]) % 360)
            # print(symbols[(i + j)][0], symbols[(i + j + 1)][0],
            #      abs(symbols[(i + j)][0] - symbols[(i + j + 1)][0] + 360) % 360)
            if abs(symbols[(i + j)][0] - symbols[(i + j + 1)][0] + 360) % 360 > 30:
                votes = []
                break
            else:
                votes.append(symbols[i + j])

        if len(votes) == 0:
            continue

        # print("".join(map(lambda v: v[1], votes)))
        temp_offset = necklace.find("".join(map(lambda v: v[1], votes)))

        for j in range(i, i + 5):
            total[j % (len(symbols) // 2)] = [j % (len(symbols) // 2), temp_offset]
            temp_offset += 1
            temp_offset %= 20

    # print(total)

    obj_points = []
    pln_points = []

    for i in range(20):
        if i >= len(total) or symbols[i] is None:
            break

        offset = total[i][1]
        if offset == -1:
            continue

        symbols[i][4] = offset

        angle = math.radians(18 * offset)

        pln_p = [
            (75 * np.cos(angle)),
            (75 * np.sin(angle))
        ]

        obj_points.append(symbols[i][2])
        pln_points.append(pln_p)

        cv.putText(frame, f"{offset}", symbols[i][2], cv.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 0, 0), 2, cv.LINE_AA)

    if len(obj_points) < 4 or len(pln_points) < 4:
        return

    '''
    H = cv.findHomography(np.array(obj_points), np.array(pln_points))

    transformed = cv.warpPerspective(original, H[0], (1000, 1000))

    cv.line(transformed, [0, 0], [75, 0], (255, 0, 0), 5)
    cv.line(transformed, [0, 0], [0, 75], (0, 255, 0), 5)
    cv.line(transformed, [0, 0], [-75, 0], (255, 255, 0), 5)
    cv.line(transformed, [0, 0], [0, -75], (0, 0, 255), 5)
    cv.drawMarker(transformed, [0, 0], (0, 0, 0), cv.MARKER_STAR, thickness=5)
    '''
    # obj_points = list(map(lambda i: list(i).append(0), obj_points))

    obj_points = np.array(obj_points, dtype=np.float32)
    pln_points = np.array(pln_points, dtype=np.float32)

    pln_points = np.append(pln_points, np.zeros((len(pln_points), 1)), axis=1)

    _, r, t = cv.solvePnP(np.array(pln_points, dtype=np.float32), np.array(obj_points, dtype=np.float32),
                          mtx, dist, flags=cv.SOLVEPNP_IPPE)

    # r = cv.Rodrigues(r)
    # print(r)
    # print(t)

    g, _ = cv.projectPoints(np.array([
        [0, 0, 0],
        [75, 0, 0],
        [0, 75, 0],
        [0, -75, 0],
        [-75, 0, 0],
        [0, 0, 75],

        # [800, 800, 0],
        # [800, 200, 0],
        # [200, 200, 0],
        # [200, 800, 0],
    ], dtype=np.float32), r, t, mtx, dist)
    # print("g (" + str(g[1][0]) + ")")
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[1][0]], (255, 0, 0), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[2][0]], (0, 255, 0), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[3][0]], (0, 0, 255), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[4][0]], (255, 255, 0), 5)
    cv.line(frame, [round(i) for i in g[0][0]], [round(i) for i in g[5][0]], (255, 0, 255), 5)

    # cv.line(frame, [round(i) for i in g[1][0]], [round(i) for i in g[7][0]], (255, 0, 255), 5)
    # cv.line(frame, [round(i) for i in g[2][0]], [round(i) for i in g[6][0]], (255, 0, 255), 5)
    # cv.line(frame, [round(i) for i in g[3][0]], [round(i) for i in g[8][0]], (255, 0, 255), 5)
    # cv.line(frame, [round(i) for i in g[9][0]], [round(i) for i in g[4][0]], (255, 0, 255), 5)

    cv.drawMarker(frame, [round(i) for i in g[0][0]], (255, 0, 255), cv.MARKER_STAR)

    # test = [500, 500, 0]

    # cv.imshow('transformed', transformed)

    return center, r, t
